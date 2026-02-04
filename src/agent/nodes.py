import os
import streamlit as st
from typing import List, Literal, Optional, Any, Dict
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range
from fastembed import TextEmbedding
from supabase import create_client, Client
from src.agent.state import AgentState, ShoppingCartItem

load_dotenv()

# Initialize external services securely
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource(show_spinner=False)
def get_qdrant_client():
    # Use in-memory Qdrant to avoid multiprocess file lock errors on Streamlit Cloud
    return QdrantClient(location=":memory:")

embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")

# Hardcode temperature 0.0 for deterministic tool outputs
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)

# --- Define the Structured Output Schema ---
class SearchParams(BaseModel):
    query: str = Field(default="", description="The semantic search query without apostrophes or special characters. E.g., 'mens hat'.")
    max_price_inr: float = Field(default=0.0, description="Maximum price constraint in INR. 0.0 means no limit.")
    sort_by: Literal["relevance", "price_asc", "price_desc"] = Field(default="relevance", description="Extract if user asks for cheapest or most expensive.")

class RouterOutput(BaseModel):
    action: Literal["search", "chat"] = Field(description="The next action to take.")
    search_params: SearchParams = Field(default_factory=SearchParams, description="Populate only if action is 'search'.")
    response: str = Field(default="", description="The message to display to the user if action is 'chat'. Do not use apostrophes or quotes.")

# --- LangGraph Nodes ---

def supervisor_node(state: AgentState) -> dict:
    """Analyzes the user's latest input and decides to Search or Chat."""
    messages = state["messages"]
    
    system_prompt = """
    You are an intelligent AI Shopping Assistant for an Indian fashion e-commerce store.
    Analyze the user's latest message and route them appropriately.
    - If they say "hello", route to 'chat' and say hi.
    - If they are looking to buy something like "I want a jacket under 100 rs", route to 'search' and extract the constraints.
    CRITICAL INSTRUCTION: NEVER use apostrophes, quotes, or special characters in your JSON output strings to prevent parsing errors. Use 'mens' instead of 'men's'.
    """
    
    structured_llm = llm.with_structured_output(RouterOutput)
    # The LLM evaluates the entire conversation history
    response: RouterOutput = structured_llm.invoke([SystemMessage(content=system_prompt)] + messages)
    
    if response.action == "search" and response.search_params:
        # Update search filters in LangGraph State
        filters = response.search_params.model_dump(exclude_none=True)
        return {
            "active_search_filters": filters,
            # We don't append a message here; we let the search node handle it.
            # But we can pass a routing thought inside the state if we want.
        }
    elif response.action == "chat":
        return {
            "messages": [AIMessage(content=response.response)]
        }
    return {}

def search_node(state: AgentState) -> dict:
    """Executes the Qdrant Hybrid Search based on active_search_filters."""
    filters = state.get("active_search_filters", {})
    query_text = filters.get("query", "")
    max_price = filters.get("max_price_inr", 0.0)
    sort_by = filters.get("sort_by", "relevance")
    
    # 1. Embed Query
    if query_text:
        query_vector = list(embedding_model.embed([query_text]))[0].tolist()
    else:
        # Fallback to random/generic vector
        query_vector = [0.0]*384
        
    # 2. Build Qdrant Hard Filters (Hybrid Search Mechanics)
    must_conditions = []
    if max_price > 0.0:
        must_conditions.append(
            FieldCondition(
                key="price_inr",
                range=Range(lte=max_price)
            )
        )
    
    query_filter = Filter(must=must_conditions) if must_conditions else None
    
    try:
        # 3. Perform Vector Search inside local Qdrant
        qdrant = get_qdrant_client()
        hits = qdrant.query_points(
            collection_name="ecommerce_products",
            query=query_vector,
            query_filter=query_filter,
            limit=20 if sort_by != "relevance" else 5
        ).points
        
        # 3.5 Process custom sorting
        if sort_by == "price_asc":
            hits = sorted(hits, key=lambda x: x.payload["price_inr"])[:5]
        elif sort_by == "price_desc":
            hits = sorted(hits, key=lambda x: x.payload["price_inr"], reverse=True)[:5]
            
    except Exception as e:
        return {"messages": [AIMessage(content=f"Database Search Error: {str(e)}")]}
    
    if not hits:
        return {"messages": [AIMessage(content=f"I couldn't find any products matching your strict criteria (e.g., price <= {max_price} INR). Try increasing your budget!")]}
    
    # 4. Format Results
    result_text = "Here are the top matches I found:\n"
    for i, hit in enumerate(hits):
        payload = hit.payload
        result_text += f"\n{i+1}. **{payload['title']}** - ₹{payload['price_inr']}"
        result_text += f"\n   *ID: {hit.id}*"
        
    return {"messages": [AIMessage(content=result_text)]}

import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage
from qdrant_client.models import PointStruct, VectorParams, Distance
from supabase import create_client, Client
from fastembed import TextEmbedding

# Import our backend logic
from src.agent.graph import app_graph
from src.agent.state import AgentState
from src.agent.nodes import get_qdrant_client

st.set_page_config(page_title="Shopping AI Agent", page_icon="🛍️")
st.title("🛍️ Shopping AI Agent")
st.markdown("Powered by **LangGraph**, **Groq**, **Supabase**, and **Qdrant**")

@st.cache_resource(show_spinner=False)
def initialize_vector_db():
    """
    Streamlit Cloud is 'Ephemeral' (no hard drive). It boots up empty.
    This startup routine checks if Qdrant is missing the vector data.
    If it is, it pulls your persistent rows from Supabase Cloud, embeds them,
    and caches them in the Streamlit Cloud RAM/Local Qdrant instance.
    """
    COLLECTION_NAME = "ecommerce_products"
    qdrant = get_qdrant_client()
    
    if not qdrant.collection_exists(COLLECTION_NAME):
        with st.spinner("First boot detected! Hydrating local Vector DB from Supabase... (Takes ~10 seconds)"):
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            
            SUPABASE_URL = os.environ.get("SUPABASE_URL")
            SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
            
            if SUPABASE_URL and SUPABASE_KEY:
                supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
                response = supabase.table("products").select("*").execute()
                products = response.data
                
                if products:
                    embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")
                    docs = [f"{p['title']} - {p['description']}" for p in products]
                    metadata = [{"id": p["id"], "title": p["title"], "price_inr": p["price_inr"], "category": p["category"]} for p in products]
                    ids = [p["id"] for p in products]
                    
                    embeddings = list(embedding_model.embed(docs))
                    points = [
                        PointStruct(id=ids[i], vector=embeddings[i].tolist(), payload=metadata[i])
                        for i in range(len(docs))
                    ]
                    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return True

# Ensure vector DB is hydrated from Supabase if we are running fresh on the cloud
initialize_vector_db()

# Initialize session state for UI history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Accept user input
if prompt := st.chat_input("Ask me about a product (e.g., 'Find a Jacket under 100 INR')"):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)

    with st.spinner("Agents are coordinating (Routing -> Qdrant -> Supabase)..."):
        initial_state: AgentState = {
            "messages": st.session_state.messages,
            "cart": [],
            "active_search_filters": {}
        }
        final_state = app_graph.invoke(initial_state)
        ai_response = final_state["messages"][-1]
        
    with st.chat_message("assistant"):
        st.markdown(ai_response.content)
        
    st.session_state.messages.append(ai_response)

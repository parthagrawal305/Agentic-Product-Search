from typing import Annotated, Sequence, TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel

class ShoppingCartItem(BaseModel):
    product_id: str
    title: str
    price_inr: float
    quantity: int = 1

class AgentState(TypedDict):
    """
    The state of the shopping agent conversation.
    - messages: Uses add_messages to append new LLM/Human interactions to the conversation context.
    - cart: A list of items the user wants to buy.
    - active_search_filters: A dictionary containing exact constraints applied to the vector search (e.g., {'max_price_inr': 100}).
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    cart: List[ShoppingCartItem]
    active_search_filters: Dict[str, Any]

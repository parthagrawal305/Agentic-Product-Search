import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.agent.graph import app_graph
from src.agent.state import AgentState

st.set_page_config(page_title="Shopping AI Agent", page_icon="🛍️")
st.title("🛍️ Shopping AI Agent")
st.markdown("Powered by **LangGraph**, **Groq**, **Supabase**, and **Qdrant**")

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
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to state
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)

    # Invoke the LangGraph State Machine
    with st.spinner("Agents are coordinating (Routing -> Qdrant -> Supabase)..."):
        # We pass the full history to LangGraph so it natively handles Conversational Memory
        initial_state: AgentState = {
            "messages": st.session_state.messages,
            "cart": [],
            "active_search_filters": {}
        }
        
        # Execute the Graph
        final_state = app_graph.invoke(initial_state)
        
        # Extract the final AI message
        ai_response = final_state["messages"][-1]
        
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(ai_response.content)
        
    # Add assistant response to session state memory
    st.session_state.messages.append(ai_response)

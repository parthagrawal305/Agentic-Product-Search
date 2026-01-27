import os
from langchain_core.messages import HumanMessage
from src.agent.graph import app_graph
from src.agent.state import AgentState

def run_test():
    print("Test 1: Normal Greeting")
    initial_state: AgentState = {
        "messages": [HumanMessage(content="Hello there!")],
        "cart": [],
        "active_search_filters": {}
    }
    result = app_graph.invoke(initial_state)
    print("AI:", result["messages"][-1].content)
    print("-" * 50)

    print("Test 2: Complex Search ('Find me a jacket under 100 INR')")
    initial_state: AgentState = {
        "messages": [HumanMessage(content="Find me a jacket under 100 INR")],
        "cart": [],
        "active_search_filters": {}
    }
    result = app_graph.invoke(initial_state)
    print("AI:", result["messages"][-1].content)
    
if __name__ == "__main__":
    run_test()

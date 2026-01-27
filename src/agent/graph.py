from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import supervisor_node, search_node

# Define Conditional Routing Logic
def route_after_supervisor(state: AgentState) -> str:
    """
    Decides the next node based on whether the supervisor populated search filters.
    If the supervisor decided to 'chat', it will have populated 'messages' instead.
    """
    # If a filter was just applied by the supervisor, go to search
    if "active_search_filters" in state and state["active_search_filters"]:
        return "search"
    return "end"

# Initialize StateGraph
graph_builder = StateGraph(AgentState)

# Add Nodes
graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("search", search_node)

# Add Edges
graph_builder.set_entry_point("supervisor")

graph_builder.add_conditional_edges(
    "supervisor",
    route_after_supervisor,
    {
        "search": "search",
        "end": END
    }
)

graph_builder.add_edge("search", END)

# Compile the Graph
# Note: In a production environment, we would pass a ThreadSaver checkpointer here to persist memory across sessions!
app_graph = graph_builder.compile()

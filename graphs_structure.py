from agents import RAGState
from graphs_node import plan_node, retrieval_node, web_search_node, rerank_node, \
                    compression_node, reflection_node, final_answer_node, get_past_context_str, \
                    initialize_graphs_node, route_next_step_node
from tools import rerank_documents_function, web_search_function
from graphs_node import policy_agent

from graphs_node import route_by_tool, should_continue_node


# Initialize the graphs_node module with the required functions and agents
initialize_graphs_node(
    web_search_function, 
    rerank_documents_function, 
    policy_agent
)

# Building the StateGraph: Wiring the Deep Thinking RAG Machine
from langgraph.graph import StateGraph, END
def build_graph():
        
    graph = StateGraph(RAGState)

    # Add nodes to the graph 
    # add_node parameters: node_name, node_function
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve_10k", retrieval_node)
    graph.add_node("retrieve_web", web_search_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("compress", compression_node)
    graph.add_node("reflect", reflection_node)
    graph.add_node("route_next_step", route_next_step_node)  # New routing node
    graph.add_node("generate_final_answer", final_answer_node)

    # Define edges
    graph.set_entry_point("plan")
    graph.add_conditional_edges(
        "plan",
        route_by_tool,
        {
            "search_10k": "retrieve_10k",
            "search_web": "retrieve_web",
        },
    )
    graph.add_edge("retrieve_10k", "rerank")
    graph.add_edge("retrieve_web", "rerank")
    graph.add_edge("rerank", "compress")
    graph.add_edge("compress", "reflect")
    graph.add_conditional_edges(
        "reflect",
        should_continue_node,
        {
            "continue": "route_next_step", # Route to next step instead of plan
            "finish": "generate_final_answer",
        },
    )
    graph.add_conditional_edges(
        "route_next_step",
        route_by_tool,
        {
            "search_10k": "retrieve_10k",
            "search_web": "retrieve_web",
        },
    )
    graph.add_edge("generate_final_answer", END)

    print("StateGraph constructed successfully.")

    return graph
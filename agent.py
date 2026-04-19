"""
Research → Summarize Agent using LangGraph
A simple agent that researches a query and then summarizes the findings.
"""

from typing import Any, Dict
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
import asyncio

# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """Shared state between nodes"""
    query: str
    research: str
    summary: str


# ============================================================================
# NODE IMPLEMENTATIONS
# ============================================================================

async def research_node(state: AgentState) -> AgentState:
    """
    Research Node: Takes a query and fetches information
    (Mocked for now - can be replaced with real API calls)
    """
    query = state["query"]
    
    # Mock research data - replace with real API calls later
    # Options: SerpAPI, Tavily, DuckDuckGo API, etc.
    mock_research = f"""
    Research Results for: "{query}"
    
    This is detailed information about {query}. 
    Key findings:
    1. Point one about the topic
    2. Point two about the topic
    3. Point three about the topic
    
    Sources: Wikipedia, academic databases, recent publications
    """
    
    print(f"🔍 Research Node: Researching '{query}'...")
    state["research"] = mock_research.strip()
    
    return state


async def summarize_node(state: AgentState) -> AgentState:
    """
    Summarize Node: Takes research and produces a concise summary
    Uses an LLM to generate the summary
    """
    research = state["research"]
    
    # For production, uncomment this and set your API key:
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-4", api_key="YOUR_API_KEY")
    # 
    # prompt = f"Summarize this in 3-4 lines:\n{research}"
    # result = await llm.ainvoke(prompt)
    # state["summary"] = result.content
    
    # Mock summary for testing (replace with real LLM)
    mock_summary = f"""
    Summary: This topic is significant because it combines multiple 
    important concepts. The key takeaway is that understanding this helps 
    in practical applications. Further research is recommended.
    """
    
    print(f"✍️  Summarize Node: Creating summary...")
    state["summary"] = mock_summary.strip()
    
    return state


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_graph():
    """Create and compile the agent graph"""
    
    # Initialize the graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("research", research_node)
    graph.add_node("summarize", summarize_node)
    
    # Add edges (flow)
    graph.add_edge(START, "research")
    graph.add_edge("research", "summarize")
    graph.add_edge("summarize", END)
    
    # Compile the graph
    app = graph.compile()
    
    return app


# ============================================================================
# EXECUTION
# ============================================================================

async def run_agent(query: str) -> Dict[str, Any]:
    """Run the agent with a given query"""
    
    # Create the agent
    agent = create_graph()
    
    # Initial state
    initial_state = {
        "query": query,
        "research": "",
        "summary": ""
    }
    
    print(f"\n{'='*70}")
    print(f"🚀 Starting Research → Summarize Agent")
    print(f"{'='*70}")
    print(f"Query: {query}\n")
    
    # Run the agent
    result = await agent.ainvoke(initial_state)
    
    # Display results
    print(f"\n{'-'*70}")
    print(f"📊 RESEARCH FINDINGS:")
    print(f"{'-'*70}")
    print(result["research"])
    
    print(f"\n{'-'*70}")
    print(f"📝 SUMMARY:")
    print(f"{'-'*70}")
    print(result["summary"])
    print(f"{'='*70}\n")
    
    return result


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Example queries to test
    queries = [
        "What is quantum computing?",
        "How does machine learning work?",
    ]
    
    # Run the agent
    for query in queries:
        result = asyncio.run(run_agent(query))
        print()

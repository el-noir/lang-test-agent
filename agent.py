"""
Enhanced version with real LLM integration
Supports OpenAI, Anthropic, and other LangChain-compatible LLMs
"""

from typing import Any, Dict
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_groq import ChatGroq

import asyncio
import os

# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """Shared state between nodes"""
    query: str
    research: str
    summary: str


# ============================================================================
# LLM INITIALIZATION (Choose one)
# ============================================================================

def get_llm():
    """
    Initialize LLM provider (required - no fallback)
    Tries providers in order: OpenAI, Anthropic, Ollama, Groq
    """
    
    # Try OpenAI
    try:
        from langchain_openai import ChatOpenAI
        if os.getenv("OPENAI_API_KEY"):
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    except ImportError:
        pass
    
    # Try Anthropic Claude
    try:
        from langchain_anthropic import ChatAnthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            return ChatAnthropic(model="claude-3-sonnet-20240229")
    except ImportError:
        pass
    
    # Try Groq
    try:
        from langchain_groq import ChatGroq
        if os.getenv("GROQ_API_KEY"):
            return ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
    except ImportError:
        pass
    
    # Try Ollama (local)
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(model="mistral")  # No key needed
    except ImportError:
        pass
    
    # No provider available
    raise RuntimeError(
        "\n❌ ERROR: No LLM provider configured!\n"
        "Please install and configure one:\n"
        "  • OpenAI: pip install langchain-openai && export OPENAI_API_KEY=sk-...\n"
        "  • Claude: pip install langchain-anthropic && export ANTHROPIC_API_KEY=sk-ant-...\n"
        "  • Groq: pip install langchain-groq && export GROQ_API_KEY=...\n"
        "  • Ollama: pip install langchain-ollama && ollama serve && ollama pull mistral"
    )


# ============================================================================
# REAL API INTEGRATION (Uncomment to use real search)
# ============================================================================

async def fetch_real_research(query: str) -> str:
    """
    Fetch real research using Tavily API
    Required - no fallback to mock data
    """
    
    try:
        from tavily import AsyncTavilyClient
    except ImportError:
        raise RuntimeError(
            "\n❌ ERROR: Tavily not installed!\n"
            "Install with: pip install tavily-python"
        )
    
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "\n❌ ERROR: TAVILY_API_KEY not set!\n"
            "Get free API key at: https://tavily.com/\n"
            "Then set: export TAVILY_API_KEY=your_key_here"
        )
    
    client = AsyncTavilyClient(api_key=api_key)
    response = await client.search(query=query, max_results=5)
    
    # Extract and format results
    results = []
    for item in response["results"]:
        results.append(f"- {item['title']}: {item['content']}")
    
    if not results:
        raise RuntimeError(f"No search results found for: {query}")
    
    return "\n".join(results)


# ============================================================================
# NODE IMPLEMENTATIONS
# ============================================================================

async def research_node(state: AgentState) -> AgentState:
    """Research Node: Fetches real information via Tavily API"""
    
    query = state["query"]
    print(f"🔍 Research Node: Searching for '{query}'...")
    
    research = await fetch_real_research(query)
    state["research"] = research
    return state


async def summarize_node(state: AgentState) -> AgentState:
    """Summarize Node: Creates a concise summary using real LLM"""
    
    research = state["research"]
    llm = get_llm()  # Will raise error if no provider available
    
    print(f"✍️  Summarize Node: Creating summary...")
    
    # Prepare prompt
    prompt = f"""Summarize the following in 3-4 concise sentences:

{research}

Summary:"""
    
    # Call LLM (no fallback)
    result = await llm.ainvoke(prompt)
    state["summary"] = result.content
    
    return state


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_graph():
    """Create and compile the agent graph"""
    
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("research", research_node)
    graph.add_node("summarize", summarize_node)
    
    # Add edges
    graph.add_edge(START, "research")
    graph.add_edge("research", "summarize")
    graph.add_edge("summarize", END)
    
    return graph.compile()


# ============================================================================
# EXECUTION
# ============================================================================

async def run_agent(query: str) -> Dict[str, Any]:
    """Run the agent"""
    
    agent = create_graph()
    
    initial_state = {
        "query": query,
        "research": "",
        "summary": ""
    }
    
    print(f"\n{'='*70}")
    print(f"🚀 Research → Summarize Agent")
    print(f"{'='*70}")
    print(f"Query: {query}\n")
    
    result = await agent.ainvoke(initial_state)
    
    print(f"\n{'-'*70}")
    print(f"📝 SUMMARY:")
    print(f"{'-'*70}")
    print(result["summary"])
    print(f"{'='*70}\n")
    
    return result


# ============================================================================
# DEPLOYMENT ENTRYPOINT
# ============================================================================

async def run(payload: Dict[str, Any], ctx=None) -> Dict[str, Any]:
    """
    Deployment entrypoint for agent platforms
    Expected signature: run(payload, context)
    """
    query = payload.get("query", "What is quantum computing?")
    return await run_agent(query)


# ============================================================================
# SETUP INSTRUCTIONS
# ============================================================================

SETUP_INSTRUCTIONS = """
🚀 Required Setup (No Mocking):

1. Install dependencies:
   pip install -r requirements.txt

2. Set up REQUIRED APIs:

   🔍 WEB SEARCH (Required):
   - pip install tavily-python
   - Get free key: https://tavily.com/
   - export TAVILY_API_KEY=your_key_here

   🤖 LLM PROVIDER (Pick one):
   
   Option A - OpenAI:
   $ pip install langchain-openai
   $ export OPENAI_API_KEY=sk-...
   
   Option B - Anthropic Claude:
   $ pip install langchain-anthropic
   $ export ANTHROPIC_API_KEY=sk-ant-...
   
   Option C - Groq (fastest, free tier):
   $ pip install langchain-groq
   $ export GROQ_API_KEY=...
   
   Option D - Local Ollama (free, offline):
   $ pip install langchain-ollama
   $ ollama serve        # In terminal 1
   $ ollama pull mistral # In terminal 2

3. Run:
   python agent_enhanced.py

NOTE: Agent will fail with clear error messages if APIs not configured!
"""

if __name__ == "__main__":
    print(SETUP_INSTRUCTIONS)
    
    # Test queries
    queries = [
        "What is quantum computing?",
        "Explain machine learning",
    ]
    
    for query in queries:
        result = asyncio.run(run_agent(query))

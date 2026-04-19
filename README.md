# Research → Summarize Agent with LangGraph

A lightweight agent that performs web search and generates summaries using LLMs.

## 📁 Files

- **`agent.py`** - Basic mock version (testing, no API keys)
- **`agent_enhanced.py`** - Production version with real web search + LLM
- **`requirements.txt`** - Dependencies

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

**Tavily Search (Free):**
```bash
# Get key at: https://tavily.com/
export TAVILY_API_KEY=your_key_here
```

**LLM Provider (choose one):**

OpenAI:
```bash
export OPENAI_API_KEY=sk-...
```

Anthropic:
```bash
pip install langchain-anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

Local Ollama (Free!):
```bash
# Install from: https://ollama.ai
ollama serve  # In one terminal
ollama pull mistral  # In another
pip install langchain-ollama
```

### 3. Run the Agent

```bash
# With mock data (testing)
python agent.py

# With real web search
python agent_enhanced.py
```

## 🏗️ Architecture

```
Query
  ↓
[Research] → Real web search via Tavily
  ↓
[Summarize] → LLM generates concise summary
  ↓
Answer
```

## ⚙️ Customization

### Change LLM Provider

Edit `agent_enhanced.py`, function `get_llm()`:

```python
# OpenAI (default)
from langchain_openai import ChatOpenAI
return ChatOpenAI(model="gpt-3.5-turbo")

# Claude
from langchain_anthropic import ChatAnthropic
return ChatAnthropic(model="claude-3-sonnet-20240229")

# Local Ollama
from langchain_ollama import ChatOllama
return ChatOllama(model="mistral")
```

## 📚 Resources

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Tavily Search](https://tavily.com/)
- [OpenAI API](https://platform.openai.com/)
- [Ollama](https://ollama.ai)

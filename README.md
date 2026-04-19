# Human-in-the-Loop Agentic AI Chatbot with Web Search

A sophisticated **Human-in-the-Loop (HITL)** chatbot system powered by **LangGraph**, **LangChain**, and **Google Gemini**, featuring intelligent web search integration, confidence-based response validation, and interactive user approval workflows.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## Features

**Core HITL Capabilities:**
- Human-in-the-loop validation workflow with state graph execution
- Multi-step response regeneration with rejection tracking
- Configurable prompt templates for different response formats
- Support for multiple LLM models (Gemini 1.5, 2.0, 3.0-flash)

**Web Search Integration:**
- Multi-provider web search (Tavily, Google Search, DuckDuckGo)
- Intelligent confidence scoring system
- Automatic web search triggering for low-confidence responses
- Verbose apology detection and suppression
- Clean text extraction (removes images, ads, hyperlinks)

**Response Management:**
- Dynamic output parsing (Pydantic, Comma-separated lists)
- Flexible state management with class injection
- HTML/image/URL cleaning for text-only results
- Streamlit web UI for interactive chat

**Smart Features:**
- 50+ keyword pattern matching for uncertainty detection
- Real-time data query detection
- Multi-factor confidence analysis
- Graceful fallback between search providers

---

## Architecture

### System Flow

```
User Query
    ↓
LLM Response (initial)
    ↓
High Confidence (>0.5)? 
    ├─ YES → Final Output (show LLM response)
    │
    └─ NO → Web Search
             ↓
         Refine Response ← LLM receives: original response + search results
             ↓ 
         LLM combines both & generates improved answer
             ↓
         Final Output (show refined response)
```

### Component Overview

**chatbot.py**
- Main LangGraph orchestration
- State management and transitions
- Web search node execution
- Response generation with confidence scoring

**Utils/web_search_tool.py**
- Multi-provider search abstraction
- Tavily, Google Search, DuckDuckGo implementations
- Text cleaning and HTML removal
- Result formatting

**Utils/confidence_checker.py**
- Confidence scoring engine
- 50+ uncertainty keyword patterns
- Verbose apology detection
- Real-time data requirement analysis

**Utils/prompt_manager.py**
- 5 customizable prompt templates
- Confidence-aware prompts
- Research hints for web search
- Structured response formats

**Utils/utils.py**
- Pydantic state/output models
- LLM initialization (Google Gemini)
- Output parser definitions

**streamlit_api.py**
- Web UI with Streamlit
- Accept/Reject buttons
- Result display
- Session state management

---

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- API keys (optional, based on search providers)

### Step 1: Clone and Setup

```bash
git clone <repository-url>
cd LLMbot
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install langchain langchain-google-genai langgraph pydantic streamlit tavily-python ddgs python-dotenv
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Google Gemini API (Required)
GOOGLE_API_KEY=your_google_api_key_here

# Web Search Providers (Optional - at least one recommended)
TAVILY_API_KEY=your_tavily_api_key_here
SERPAPI_API_KEY=your_serpapi_key_here

# Streamlit Configuration (Optional)
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
```

### Getting API Keys

**Google Gemini API:**
1. Go to [Google AI Studio](https://aistudio.google.com)
2. Click "Get API Key"
3. Create new API key for project

**Tavily Search API:**
1. Visit [Tavily AI](https://tavily.com)
2. Sign up and get API key
3. Add to `.env` (Recommended - best search quality)

**DuckDuckGo:**
- Free, no API key needed
- Installed via `pip install ddgs`

---

## Project Structure

```
LLMbot/
├── chatbot.py                    # Main LangGraph orchestration
├── streamlit_api.py              # Streamlit web UI
├── hitl_chatbot.py               # Alternative HITL implementation
├── human_in_the_loop.ipynb       # Jupyter notebook demo
├── README.md                     # This file
│
├── Utils/
│   ├── web_search_tool.py       # Multi-provider web search
│   ├── confidence_checker.py    # Confidence scoring engine
│   ├── prompt_manager.py        # Prompt templates
│   ├── output_parser.py         # Output parsing logic
│   └── utils.py                 # State models and LLM init
│
└── .env                         # Configuration (create this)
```

---

## Usage Guide

### Running the Streamlit App

```bash
streamlit run streamlit_api.py
```

The app opens at `http://localhost:8501`

**UI Workflow:**
1. Enter query in text box
2. Click "Submit"
3. View LLM response with confidence score
4. Click "Accept" to approve or "Reject" to regenerate
5. If confidence is low, web search results appear automatically

### Using Chatbot Directly (Python)

```python
from chatbot import GraphForHumanInTheLoop
from Utils.utils import QueryStateForString, LLMOutputOnlyString

# Initialize chatbot
chatbot = GraphForHumanInTheLoop(
    state=QueryStateForString,
    opstate=LLMOutputOnlyString
)

# Get response
result = chatbot.execuete_graph(user_query="What is the capital of France?")
print(result)
```

### Using in Jupyter Notebook

```python
# See human_in_the_loop.ipynb for examples
from chatbot import GraphForHumanInTheLoop
from Utils.utils import QueryStateForString, LLMOutputOnlyString

# Interactive workflow with manual approval
chatbot = GraphForHumanInTheLoop(
    state=QueryStateForString,
    opstate=LLMOutputOnlyString
)

response = chatbot.execuete_graph(user_query="Your question here")
```

---

## API Reference

### WebSearchTool

```python
from Utils.web_search_tool import WebSearchTool

# Initialize
search = WebSearchTool()  # Auto-selects best provider

# Search
results = search.search("your query", num_results=5)
# Returns: [{'title': str, 'url': str, 'snippet': str}, ...]

# Format for display
formatted = search.format_search_results(results)
print(formatted)
```

### ConfidenceChecker

```python
from Utils.confidence_checker import ConfidenceChecker

# Check response confidence
response_text = "I can provide you the answer..."
confidence_score, needs_web_search = ConfidenceChecker.check_response_confidence(response_text)

# confidence_score: float 0-1 (higher = more confident)
# needs_web_search: bool (True if <0.5 or uncertainty detected)

# Detect verbose apologies
is_verbose = ConfidenceChecker.is_verbose_apology(response_text)

# Clean verbose response
cleaned = ConfidenceChecker.clean_verbose_response(response_text)
```

### PromptTemplates

```python
from Utils.prompt_manager import PromptTemplates

# Available templates:
template1 = PromptTemplates.chat_prompt_template()
template2 = PromptTemplates.confidence_aware_prompt_template()
template3 = PromptTemplates.research_hint_prompt_template()
template4 = PromptTemplates.structured_response_prompt_template()
template5 = PromptTemplates.comma_seperated_prompt_template()

# Use in chain
messages = template.format(query="Your question", format_instructions="...")
```

### State Models

```python
from Utils.utils import QueryStateForString, QueryState, LLMOutputOnlyString, LLMOutput

# String output state
state = QueryStateForString(
    query="User question",
    output="LLM response",
    confidence_score=0.85,
    needs_web_search=False,
    web_search_results=None,
    human_approval="yes",
    rejection_count=0
)

# Structured output state
state = QueryState(
    query="User question",
    output=["item1", "item2"],
    web_search_results="Search results...",
    human_approval="yes"
)
```

---

## Examples

### Example 1: Basic Query

**Query:** "What is photosynthesis?"

**Flow:**
1. LLM generates response with high confidence
2. No web search triggered (confidence > 0.5)
3. Response shown to user for approval

### Example 2: Real-time Data Query

**Query:** "What is today's weather in New York?"

**Flow:**
1. LLM detects real-time data requirement
2. Confidence score drops (< 0.5)
3. Web search automatically triggered
4. Latest weather data retrieved and displayed
5. Verbose LLM apologies suppressed

### Example 3: Uncertain Query

**Query:** "Can you explain the latest AI developments?"

**Flow:**
1. LLM response contains uncertainty keywords ("cannot provide accurate", "knowledge is based on")
2. Confidence scorer detects 50+ patterns
3. Web search triggered for current information
4. Combined results shown: LLM + web search
5. User approves or rejects

### Example 4: Multi-step Rejection

**Flow:**
1. User asks question
2. Rejects LLM response (rejection_count: 1)
3. System regenerates answer
4. Still not satisfied (rejection_count: 2)
5. Eventually approved with tracking

---

## Troubleshooting

### Issue: "GOOGLE_API_KEY not found"

**Solution:**
```bash
# Create .env file
echo GOOGLE_API_KEY=your_key_here > .env

# Or set environment variable
set GOOGLE_API_KEY=your_key_here  # Windows
export GOOGLE_API_KEY=your_key_here  # Linux/macOS
```

### Issue: Web search returns 0 results

**Causes & Solutions:**
- Query too specific: Try more general terms
- No API keys configured: Add TAVILY_API_KEY to .env
- All providers down: Restart and try again

```python
# Debug: Check configured provider
from Utils.web_search_tool import WebSearchTool
search = WebSearchTool()
print(f"Using provider: {search.provider}")
```

### Issue: Streamlit app not loading

**Solutions:**
```bash
# Clear cache
streamlit cache clear

# Run with verbose output
streamlit run streamlit_api.py --logger.level=debug

# Check port availability
netstat -ano | findstr :8501  # Windows
lsof -i :8501  # Linux/macOS
```

### Issue: LLM response format error

**Solutions:**
- Ensure Pydantic models match LLM output
- Check `Utils/utils.py` for model definitions
- Verify `output_parser.py` format instructions
- Review prompt templates in `prompt_manager.py`

### Issue: Web search timeout

**Solution:**
```python
# Increase timeout in web_search_tool.py
client = TavilyClient(api_key=key, timeout=30)  # 30 seconds
```

---

## Performance Tips

1. **Cache Results:** Implement caching for repeated queries
2. **Batch Processing:** Process multiple queries in parallel
3. **Model Selection:** Use lighter models (e.g., gemini-3-flash) for speed
4. **Search Limits:** Reduce `num_results` from 10 to 5 for faster search
5. **Provider Priority:** Use Tavily first (faster than DuckDuckGo)

---

## Future Enhancements

- Database integration for conversation history
- User authentication and session management
- Multi-language support
- Custom confidence thresholds
- Response caching system
- Advanced analytics dashboard
- RAG (Retrieval-Augmented Generation) integration

---

## License

This project is open source and available under the MIT License.

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [Examples](#examples)
3. Check `.env` configuration
4. Verify API keys are valid

---

## Version History

**v1.0.0** (April 2026)
- Initial release with HITL workflow
- Web search integration (Tavily, Google, DuckDuckGo)
- Confidence-based response validation
- Streamlit UI
- 50+ uncertainty patterns
- Text cleaning and HTML removal


    pip install langchain langgraph google-generativeai pydantic

### Notes

    Prompt templates must clearly instruct the LLM to follow output format, especially when using PydanticOutputParser.

    You can switch parsers dynamically to suit different output needs (structured vs list).

    To extend, just create a new State class (e.g. DataQueryState) and plug it into the bot.
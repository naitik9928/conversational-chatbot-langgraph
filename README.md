# LangGraph + Groq Multi-Thread Chatbot

A multi-conversation AI chatbot built using LangGraph, Groq, Streamlit, and SQLite memory.

---

# Features

- Real-time streaming responses
- Persistent chat memory
- Multiple chat threads
- Conversation switching
- SQLite checkpoint storage
- Streamlit chat interface

---

# Tech Stack

| Technology | Usage |
|---|---|
| LangGraph | Workflow orchestration |
| Groq | LLM inference |
| Streamlit | Frontend UI |
| SQLite | Persistent memory |
| LangChain | Message handling |

---

# Installation

## Clone Repository

```bash
git clone <your-repo-link>
cd <repo-name>
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

---

# Run Application

```bash
streamlit run chat_groq_frontEnd.py
```

---

# Author

Built by Naitik using LangGraph + Groq + Streamlit.

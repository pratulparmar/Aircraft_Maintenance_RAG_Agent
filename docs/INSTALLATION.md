# Installation Guide

## Prerequisites

- Python 3.11 or higher
- pip package manager
- Git

## Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/airbus-rag-agent.git
cd airbus-rag-agent
```

## Step 2: Create Virtual Environment

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install all required packages including:
- Streamlit for web interface
- LangChain for RAG orchestration
- ChromaDB for vector storage
- Groq for LLM access
- Additional dependencies

## Step 4: Configure API Key

Create a `.env` file:
```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:
```
GROQ_API_KEY=gsk_your_actual_key_here
```

Get a free API key at: https://console.groq.com

## Step 5: Run the Application

Web Interface:
```bash
streamlit run app.py
```

Command Line:
```bash
python rag_agent.py
```

The web interface will open at: http://localhost:8501

## Troubleshooting

**Issue: Python not found**
- Make sure Python 3.11+ is installed
- Try using `python3` instead of `python`

**Issue: Package installation fails**
- Upgrade pip: `pip install --upgrade pip`
- Install packages individually if needed

**Issue: Streamlit won't start**
- Check if port 8501 is already in use
- Try: `streamlit run app.py --server.port 8502`

**Issue: Groq API error**
- Verify your API key is correct in `.env`
- Check you have API credits remaining

## Verify Installation

Run this test:
```bash
python -c "import streamlit; import langchain; print('Installation successful')"
```

## Next Steps

1. Upload a PDF manual using the web interface
2. Try example queries
3. Explore different modes (Standard, Checklist, Unit Conversion)
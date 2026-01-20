# app.py - Streamlit UI for A360 Technical Manual RAG Agent
import os
import streamlit as st
import time
import json
from rank_bm25 import BM25Okapi
import re
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from groq import Groq
from dotenv import load_dotenv

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="A360 RAG Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F1F5F9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    .tool-badge {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        display: inline-block;
        margin: 0.25rem;
    }
    .answer-box {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #E2E8F0;
        margin-top: 1rem;
    }
    .source-citation {
        background-color: #FEF3C7;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F59E0B;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CONFIGURATION
# ============================================

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found. Please create a .env file with your API key.")
    st.info("Get a free API key at: https://console.groq.com")
    st.stop()


# ============================================
# UTILITY CLASSES
# ============================================

class UnitConverter:
    """Engineering unit converter"""
    
    CONVERSIONS = {
        "ft-lbs_to_nm": 1.35582,
        "nm_to_ft-lbs": 0.737562,
        "in-lbs_to_nm": 0.112985,
        "nm_to_in-lbs": 8.85075,
        "psi_to_kpa": 6.89476,
        "kpa_to_psi": 0.145038,
    }
    
    @staticmethod
    def convert(value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
        conversion_key = f"{from_unit}_to_{to_unit}"
        
        if conversion_key in UnitConverter.CONVERSIONS:
            factor = UnitConverter.CONVERSIONS[conversion_key]
            converted = value * factor
            return {
                "original_value": value,
                "original_unit": from_unit,
                "converted_value": round(converted, 2),
                "converted_unit": to_unit,
                "formatted": f"{value} {from_unit} = {round(converted, 2)} {to_unit}"
            }
        return {"error": "Conversion not supported"}

class ChecklistGenerator:
    """Generate maintenance checklists"""
    
    @staticmethod
    def generate_checklist(
        procedure_name: str,
        tools: List[str],
        steps: List[str],
        safety_warnings: List[str]
    ) -> Dict[str, Any]:
        
        return {
            "procedure": procedure_name,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pre_work_requirements": {
                "safety_briefing": True,
                "tools_prepared": True,
                "safety_warnings": safety_warnings
            },
            "required_tools": [
                {"item": tool, "status": "pending"} for tool in tools
            ],
            "procedure_steps": [
                {
                    "step_number": i + 1,
                    "description": step,
                    "completed": False
                }
                for i, step in enumerate(steps)
            ],
            "sign_off": {
                "technician": None,
                "supervisor": None,
                "date": None
            }
        }

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.vectorstore = None
    st.session_state.chunks = []
    st.session_state.bm25 = None
    st.session_state.embeddings = None
    st.session_state.groq_client = None
    st.session_state.query_history = []

# ============================================
# HELPER FUNCTIONS
# ============================================

@st.cache_resource
def load_embeddings():
    """Load embedding model (cached)"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def initialize_groq():
    """Initialize Groq client"""
    if st.session_state.groq_client is None:
        st.session_state.groq_client = Groq(api_key=GROQ_API_KEY)

def process_pdf(pdf_file):
    """Process uploaded PDF"""
    
    with st.spinner("Processing PDF..."):
        # Save uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.read())
        
        # Load and process
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = load_embeddings()
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_streamlit"
        )
        
        # Create BM25 index
        chunk_texts = [chunk.page_content for chunk in chunks]
        tokenized_chunks = [text.lower().split() for text in chunk_texts]
        bm25 = BM25Okapi(tokenized_chunks)
        
        # Store in session state
        st.session_state.vectorstore = vectorstore
        st.session_state.chunks = chunks
        st.session_state.bm25 = bm25
        st.session_state.embeddings = embeddings
        st.session_state.initialized = True
        
        return len(documents), len(chunks)

def hybrid_search(question: str, k: int = 3):
    """Perform hybrid search"""
    
    vectorstore = st.session_state.vectorstore
    chunks = st.session_state.chunks
    bm25 = st.session_state.bm25
    
    # Adjust alpha
    alpha = 0.5
    if any(char.isdigit() for char in question) or any(c in question for c in ['-', '_']):
        alpha = 0.3
    
    # Semantic search
    semantic_results = vectorstore.similarity_search_with_score(question, k=k*2)
    
    if semantic_results:
        max_semantic = max(score for _, score in semantic_results)
        semantic_dict = {
            i: (chunks[i], 1 - score/max_semantic if max_semantic > 0 else 0)
            for i, (doc, score) in enumerate(semantic_results)
        }
    else:
        semantic_dict = {}
    
    # Keyword search
    tokenized_query = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    keyword_dict = {
        i: (chunks[i], bm25_scores[i] / max_bm25)
        for i in range(len(chunks))
    }
    
    # Combine
    combined = {}
    all_indices = set(semantic_dict.keys()) | set(keyword_dict.keys())
    
    for idx in all_indices:
        _, semantic_score = semantic_dict.get(idx, (None, 0))
        _, keyword_score = keyword_dict.get(idx, (None, 0))
        
        combined_score = alpha * semantic_score + (1 - alpha) * keyword_score
        combined[idx] = (chunks[idx], combined_score)
    
    sorted_results = sorted(combined.items(), key=lambda x: x[1][1], reverse=True)[:k]
    
    docs = [doc for _, (doc, score) in sorted_results]
    scores = [score for _, (doc, score) in sorted_results]
    context = "\n\n".join([doc.page_content for doc in docs])
    
    return context, docs, scores

def query_agent(question: str, mode: str = "standard"):
    """Query the agent with different modes"""
    
    initialize_groq()
    
    start_time = time.time()
    
    # Retrieve context
    context, docs, scores = hybrid_search(question)
    
    retrieval_time = time.time() - start_time
    
    # Build prompt based on mode
    if mode == "checklist":
        prompt = f"""You are an aircraft maintenance assistant. Extract the following from the context:

Context:
{context}

Question: {question}

Provide your answer AND extract:
ANSWER: [Your answer]

CHECKLIST_DATA:
PROCEDURE: [procedure name]
TOOLS: [tool1], [tool2], [tool3]
STEPS: [step1] | [step2] | [step3]
SAFETY: [warning1] | [warning2]
"""
    else:
        prompt = f"""You are an aircraft maintenance assistant.

Context:
{context}

Question: {question}

Provide a clear, concise answer based on the context."""
    
    # Get LLM response
    llm_start = time.time()
    
    try:
        response = st.session_state.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=800
        )
        
        llm_time = time.time() - llm_start
        answer = response.choices[0].message.content
        
        # Process answer based on mode
        result = {
            "answer": answer,
            "sources": docs,
            "scores": scores,
            "retrieval_time": retrieval_time,
            "llm_time": llm_time,
            "total_time": time.time() - start_time,
            "tools_used": []
        }
        
        # Handle checklist generation
        if mode == "checklist" and "CHECKLIST_DATA:" in answer:
            parts = answer.split("CHECKLIST_DATA:")
            result["answer"] = parts[0].replace("ANSWER:", "").strip()
            checklist_data = parts[1]
            
            try:
                procedure = re.search(r'PROCEDURE:\s*(.+?)(?:\n|$)', checklist_data)
                tools = re.search(r'TOOLS:\s*(.+?)(?:\n|$)', checklist_data)
                steps = re.search(r'STEPS:\s*(.+?)(?:\n|$)', checklist_data)
                safety = re.search(r'SAFETY:\s*(.+?)(?:\n|$)', checklist_data)
                
                if all([procedure, tools, steps, safety]):
                    generator = ChecklistGenerator()
                    checklist = generator.generate_checklist(
                        procedure_name=procedure.group(1).strip(),
                        tools=[t.strip() for t in tools.group(1).split(',')],
                        steps=[s.strip() for s in steps.group(1).split('|')],
                        safety_warnings=[w.strip() for w in safety.group(1).split('|')]
                    )
                    result["checklist"] = checklist
                    result["tools_used"].append("checklist_generator")
            except:
                pass
        
        # Handle unit conversion
        if any(word in question.lower() for word in ["convert", "n-m", "nm", "newton", "metric"]):
            torque_matches = re.findall(r'(\d+)\s*(ft-lbs|in-lbs)', result["answer"])
            if torque_matches:
                conversions = []
                converter = UnitConverter()
                for value, unit in torque_matches:
                    if unit == "ft-lbs":
                        conv = converter.convert(float(value), "ft-lbs", "nm")
                        conversions.append(conv)
                    elif unit == "in-lbs":
                        conv = converter.convert(float(value), "in-lbs", "nm")
                        conversions.append(conv)
                
                if conversions:
                    result["unit_conversions"] = conversions
                    result["tools_used"].append("unit_converter")
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

# ============================================
# MAIN APP
# ============================================

def main():
    
    # Header
    st.markdown('<div class="main-header">✈️ A360 Technical Manual RAG Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Aircraft Maintenance Assistant with Agentic Capabilities</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # PDF Upload
        st.subheader("1. Upload Manual")
        uploaded_file = st.file_uploader(
            "Upload PDF Manual",
            type=['pdf'],
            help="Upload an aircraft maintenance manual in PDF format"
        )
        
        if uploaded_file and not st.session_state.initialized:
            if st.button("Process PDF", type="primary"):
                pages, chunks = process_pdf(uploaded_file)
                st.success(f"Processed {pages} page(s) into {chunks} chunks")
        
        if st.session_state.initialized:
            st.success("System Ready")
            
            # Query Mode
            st.subheader("2. Query Mode")
            query_mode = st.selectbox(
                "Select Mode",
                ["Standard Q&A", "Generate Checklist", "Unit Conversion"],
                help="Choose how the agent should respond"
            )
            
            # Advanced Options
            with st.expander("Advanced Options"):
                show_sources = st.checkbox("Show source citations", value=True)
                show_metrics = st.checkbox("Show performance metrics", value=True)
                num_sources = st.slider("Number of sources", 1, 5, 3)
        
        # Info
        st.divider()
        st.caption("Built with LangChain, Groq, and Streamlit")
    
    # Main Content
    if not st.session_state.initialized:
        # Welcome screen
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🔍 Hybrid Search")
            st.write("Combines semantic and keyword search for accurate retrieval")
        
        with col2:
            st.markdown("### 🛠️ Engineering Tools")
            st.write("Unit conversion, checklist generation, safety checking")
        
        with col3:
            st.markdown("### 📊 Structured Output")
            st.write("JSON checklists ready for MES/ERP integration")
        
        st.info("👈 Upload a PDF manual in the sidebar to get started")
        
        # Example queries
        st.subheader("Example Queries")
        
        examples = [
            "What tools do I need to change a tire?",
            "List all torque specifications",
            "Generate a checklist for tire replacement",
            "Convert 450 ft-lbs to Newton-meters",
            "What are the safety warnings?"
        ]
        
        for example in examples:
            st.code(example, language=None)
    
    else:
        # Query interface
        st.subheader("Ask a Question")
        
        # Pre-filled examples
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔧 Tools Required"):
                st.session_state.current_query = "What tools do I need to change a tire?"
        
        with col2:
            if st.button("📋 Generate Checklist"):
                st.session_state.current_query = "Generate a checklist for tire replacement"
        
        with col3:
            if st.button("⚡ Torque Specs"):
                st.session_state.current_query = "List all torque specifications"
        
        # Query input
        query = st.text_input(
            "Enter your question:",
            value=st.session_state.get('current_query', ''),
            placeholder="e.g., What is the wheel nut torque specification?",
            key="query_input"
        )
        
        if st.button("Submit Query", type="primary") and query:
            
            with st.spinner("Processing query..."):
                
                # Determine mode
                mode_map = {
                    "Standard Q&A": "standard",
                    "Generate Checklist": "checklist",
                    "Unit Conversion": "standard"
                }
                mode = mode_map.get(query_mode, "standard")
                
                # Query agent
                result = query_agent(query, mode)
                
                # Display results
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    # Answer
                    st.markdown("### Answer")
                    st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
                    
                    # Tools used
                    if result.get("tools_used"):
                        st.markdown("**Tools Used:**")
                        for tool in result["tools_used"]:
                            st.markdown(f'<span class="tool-badge">{tool}</span>', unsafe_allow_html=True)
                    
                    # Unit conversions
                    if result.get("unit_conversions"):
                        st.markdown("### Unit Conversions")
                        for conv in result["unit_conversions"]:
                            st.info(conv["formatted"])
                    
                    # Checklist
                    if result.get("checklist"):
                        st.markdown("### Generated Checklist")
                        
                        # Display checklist
                        checklist = result["checklist"]
                        
                        with st.expander("View Checklist Details", expanded=True):
                            st.write(f"**Procedure:** {checklist['procedure']}")
                            st.write(f"**Generated:** {checklist['generated_at']}")
                            
                            st.write("**Required Tools:**")
                            for tool in checklist['required_tools']:
                                st.write(f"- {tool['item']}")
                            
                            st.write("**Procedure Steps:**")
                            for step in checklist['procedure_steps']:
                                st.write(f"{step['step_number']}. {step['description']}")
                        
                        # Download button
                        checklist_json = json.dumps(checklist, indent=2)
                        st.download_button(
                            label="Download Checklist (JSON)",
                            data=checklist_json,
                            file_name=f"checklist_{int(time.time())}.json",
                            mime="application/json"
                        )
                    
                    # Sources
                    if show_sources and result.get("sources"):
                        st.markdown("### Sources")
                        for i, (doc, score) in enumerate(zip(result["sources"], result["scores"]), 1):
                            with st.expander(f"Source {i} (Confidence: {score:.2f})"):
                                st.text(doc.page_content[:500] + "...")
                                st.caption(f"Page: {doc.metadata.get('page', 'N/A')}")
                    
                    # Performance metrics
                    if show_metrics:
                        st.markdown("### Performance Metrics")
                        cols = st.columns(3)
                        
                        with cols[0]:
                            st.metric("Retrieval Time", f"{result['retrieval_time']:.2f}s")
                        
                        with cols[1]:
                            st.metric("LLM Time", f"{result['llm_time']:.2f}s")
                        
                        with cols[2]:
                            st.metric("Total Time", f"{result['total_time']:.2f}s")
                    
                    # Add to history
                    st.session_state.query_history.append({
                        "query": query,
                        "answer": result["answer"],
                        "time": time.strftime("%H:%M:%S")
                    })
        
        # Query history
        if st.session_state.query_history:
            with st.expander("Query History"):
                for i, item in enumerate(reversed(st.session_state.query_history[-5:]), 1):
                    st.text(f"{item['time']} - {item['query']}")

if __name__ == "__main__":
    main()
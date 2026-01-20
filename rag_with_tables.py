# rag_with_tables.py - RAG with Table-Aware Parsing
import os
import time
from rank_bm25 import BM25Okapi
import numpy as np
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

print("Starting Table-Aware RAG System")
print("=" * 60)

print("\nLoading dependencies...")
start = time.time()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from groq import Groq
from dotenv import load_dotenv

print(f"Dependencies loaded in {time.time() - start:.1f} seconds")

# Configuration
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment. Create .env file with your key.")
groq_client = Groq(api_key=GROQ_API_KEY)
print("Groq API connected")

# ============================================
# TABLE DETECTION AND EXTRACTION
# ============================================

@dataclass
class TableData:
    """Represents an extracted table"""
    content: str
    headers: List[str]
    rows: List[List[str]]
    summary: str
    section: str
    table_type: str  # "torque", "parts", "procedure", "general"

class TableExtractor:
    """Extract and preserve table structure from text"""
    
    TABLE_PATTERNS = {
        "torque": r"torque|ft-lbs|in-lbs|n-m|nm",
        "parts": r"part number|p/n|assembly|component",
        "procedure": r"step|procedure|sequence",
        "specifications": r"specification|spec|requirement"
    }
    
    @staticmethod
    def detect_table_section(text: str) -> Optional[str]:
        """Detect if text contains a table section"""
        
        # Look for common table headers
        table_indicators = [
            r"SECTION.*:.*SPECIFICATIONS",
            r"SECTION.*:.*PART NUMBERS",
            r"SECTION.*:.*TORQUE",
            r"Table \d+",
            r"LIST OF",
        ]
        
        for pattern in table_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return "table_section"
        
        return None
    
    @staticmethod
    def extract_key_value_pairs(text: str) -> List[Dict[str, str]]:
        """Extract key-value pairs that might be in table format"""
        
        pairs = []
        
        # Pattern: "Label: Value" or "Label - Value"
        pattern = r'^([^:\n-]+?)[:–-]\s*(.+?)$'
        
        for line in text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            match = re.match(pattern, line)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                
                # Skip if too short or looks like a heading
                if len(key) > 3 and len(value) > 0:
                    pairs.append({
                        "key": key,
                        "value": value,
                        "raw": line
                    })
        
        return pairs
    
    @staticmethod
    def identify_table_type(text: str, pairs: List[Dict]) -> str:
        """Identify what type of table this is"""
        
        text_lower = text.lower()
        
        # Check content for table type
        if any(re.search(pattern, text_lower) for pattern in [r"torque", r"ft-lbs", r"n-m"]):
            return "torque"
        elif any(re.search(pattern, text_lower) for pattern in [r"part number", r"p/n", r"assembly"]):
            return "parts"
        elif any(re.search(pattern, text_lower) for pattern in [r"step \d+", r"procedure"]):
            return "procedure"
        else:
            return "general"
    
    @staticmethod
    def create_table_summary(pairs: List[Dict], table_type: str) -> str:
        """Create a natural language summary of the table"""
        
        if not pairs:
            return "Empty table"
        
        summaries = []
        
        if table_type == "torque":
            summaries.append("Torque Specifications:")
            for pair in pairs:
                summaries.append(f"  - {pair['key']}: {pair['value']}")
        
        elif table_type == "parts":
            summaries.append("Part Numbers:")
            for pair in pairs:
                summaries.append(f"  - {pair['key']}: {pair['value']}")
        
        else:
            summaries.append("Table Data:")
            for pair in pairs:
                summaries.append(f"  - {pair['key']}: {pair['value']}")
        
        return "\n".join(summaries)
    
    @classmethod
    def extract_tables(cls, text: str, section_name: str = "") -> List[TableData]:
        """Main method to extract tables from text"""
        
        tables = []
        
        # Detect if this is a table section
        is_table_section = cls.detect_table_section(text)
        
        if is_table_section:
            # Extract key-value pairs
            pairs = cls.extract_key_value_pairs(text)
            
            if pairs:
                # Identify table type
                table_type = cls.identify_table_type(text, pairs)
                
                # Create summary
                summary = cls.create_table_summary(pairs, table_type)
                
                # Extract headers and rows
                headers = ["Item", "Value"]
                rows = [[p["key"], p["value"]] for p in pairs]
                
                table = TableData(
                    content=text,
                    headers=headers,
                    rows=rows,
                    summary=summary,
                    section=section_name,
                    table_type=table_type
                )
                
                tables.append(table)
        
        return tables

# ============================================
# ENHANCED DOCUMENT PROCESSOR
# ============================================

class TableAwareDocumentProcessor:
    """Process documents while preserving table structure"""
    
    def __init__(self, chunk_size=800, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.table_extractor = TableExtractor()
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def process_documents(self, documents: List[Document]) -> Tuple[List[Document], List[TableData]]:
        """
        Process documents and extract tables separately
        
        Returns:
            Tuple of (text_chunks, extracted_tables)
        """
        
        all_tables = []
        enhanced_chunks = []
        
        for doc in documents:
            text = doc.page_content
            
            # Extract tables from this document
            tables = self.table_extractor.extract_tables(text)
            all_tables.extend(tables)
            
            # Create chunks from the document
            chunks = self.base_splitter.split_documents([doc])
            
            # Enhance chunks with table metadata
            for chunk in chunks:
                # Check if this chunk contains table data
                chunk_tables = self.table_extractor.extract_tables(chunk.page_content)
                
                if chunk_tables:
                    # Mark this chunk as containing a table
                    chunk.metadata['contains_table'] = True
                    chunk.metadata['table_type'] = chunk_tables[0].table_type
                    
                    # Add table summary to the chunk content for better retrieval
                    enhanced_content = f"{chunk.page_content}\n\n[TABLE SUMMARY]\n{chunk_tables[0].summary}"
                    chunk.page_content = enhanced_content
                
                enhanced_chunks.append(chunk)
        
        return enhanced_chunks, all_tables

# ============================================
# LOAD AND PROCESS DATA
# ============================================

print("\nLoading maintenance manual...")
start = time.time()
loader = PyPDFLoader("test_manual.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} page(s) in {time.time() - start:.1f} seconds")

print("\nProcessing with table-aware parser...")
start = time.time()
processor = TableAwareDocumentProcessor(chunk_size=800, chunk_overlap=100)
chunks, extracted_tables = processor.process_documents(documents)
chunk_texts = [chunk.page_content for chunk in chunks]
print(f"Created {len(chunks)} chunks and extracted {len(extracted_tables)} tables in {time.time() - start:.1f} seconds")

# Display extracted tables
if extracted_tables:
    print("\nExtracted Tables:")
    for i, table in enumerate(extracted_tables, 1):
        print(f"\nTable {i} - Type: {table.table_type}")
        print(f"Section: {table.section}")
        print(f"Rows: {len(table.rows)}")
        print(f"Summary:\n{table.summary[:200]}...")

print("\nLoading embedding model...")
start = time.time()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print(f"Embedding model ready in {time.time() - start:.1f} seconds")

print("\nInitializing vector database...")
start = time.time()
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db_tables"
)

if len(vectorstore.get()['ids']) == 0:
    vectorstore.add_documents(chunks)
    print(f"Vector database created in {time.time() - start:.1f} seconds")
else:
    print(f"Vector database loaded in {time.time() - start:.1f} seconds")

print("\nCreating BM25 index...")
start = time.time()
tokenized_chunks = [text.lower().split() for text in chunk_texts]
bm25 = BM25Okapi(tokenized_chunks)
print(f"BM25 index created in {time.time() - start:.1f} seconds")

print("\n" + "=" * 60)
print("TABLE-AWARE SYSTEM INITIALIZED")
print("=" * 60)

# ============================================
# TABLE-AWARE RETRIEVAL
# ============================================

def retrieve_with_table_awareness(question: str, k: int = 3) -> Dict[str, Any]:
    """Retrieve context with special handling for table queries"""
    
    # Check if query is asking for structured data
    table_query_indicators = [
        "list all", "all torque", "all parts", "all specifications",
        "table", "chart", "specifications"
    ]
    
    is_table_query = any(indicator in question.lower() for indicator in table_query_indicators)
    
    # Standard hybrid search
    alpha = 0.5
    if any(char.isdigit() for char in question) or any(c in question for c in ['-', '_']):
        alpha = 0.3
    
    semantic_results = vectorstore.similarity_search_with_score(question, k=k*2)
    
    if semantic_results:
        max_semantic = max(score for _, score in semantic_results)
        semantic_dict = {
            i: (chunks[i], 1 - score/max_semantic if max_semantic > 0 else 0)
            for i, (doc, score) in enumerate(semantic_results)
        }
    else:
        semantic_dict = {}
    
    tokenized_query = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    keyword_dict = {
        i: (chunks[i], bm25_scores[i] / max_bm25)
        for i in range(len(chunks))
    }
    
    combined = {}
    all_indices = set(semantic_dict.keys()) | set(keyword_dict.keys())
    
    for idx in all_indices:
        _, semantic_score = semantic_dict.get(idx, (None, 0))
        _, keyword_score = keyword_dict.get(idx, (None, 0))
        
        # Boost table chunks for table queries
        boost = 1.0
        if is_table_query and chunks[idx].metadata.get('contains_table'):
            boost = 1.5
        
        combined_score = (alpha * semantic_score + (1 - alpha) * keyword_score) * boost
        combined[idx] = (chunks[idx], combined_score)
    
    sorted_results = sorted(combined.items(), key=lambda x: x[1][1], reverse=True)[:k]
    
    docs = [doc for _, (doc, score) in sorted_results]
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Check if any retrieved chunks contain tables
    has_tables = any(doc.metadata.get('contains_table') for doc in docs)
    
    return {
        "context": context,
        "docs": docs,
        "has_tables": has_tables,
        "is_table_query": is_table_query
    }

# ============================================
# ENHANCED QUERY FUNCTION
# ============================================

def query_with_table_awareness(question: str):
    """Query with table-aware retrieval and formatting"""
    
    print(f"\nQuestion: {question}")
    query_start = time.time()
    
    # Retrieve with table awareness
    print("Performing table-aware retrieval...")
    retrieval = retrieve_with_table_awareness(question)
    
    if retrieval['has_tables']:
        print("Note: Retrieved content includes structured table data")
    
    # Build prompt
    prompt = f"""You are an aircraft maintenance assistant with access to documentation that includes tables and structured data.

Context (may include table summaries):
{retrieval['context']}

Question: {question}

Instructions:
- If the context includes table data, preserve the structure in your answer
- For specifications, list them clearly
- Cite specific values accurately

Answer:"""
    
    # Get LLM response
    print("Generating response...")
    llm_start = time.time()
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=600
        )
        
        llm_time = time.time() - llm_start
        answer = response.choices[0].message.content
        
        total_time = time.time() - query_start
        
        print(f"\nAnswer:")
        print(answer)
        
        if retrieval['has_tables']:
            print("\nNote: Answer based on structured table data")
        
        print(f"\nPerformance: Retrieval={total_time-llm_time:.2f}s | LLM={llm_time:.2f}s | Total={total_time:.2f}s")
        print("-" * 60)
        
        return answer
        
    except Exception as e:
        print(f"Error: {e}")
        return None

# ============================================
# DEMONSTRATIONS
# ============================================

print("\n" + "=" * 60)
print("TABLE-AWARE RETRIEVAL DEMONSTRATIONS")
print("=" * 60)

# Demo 1: List all torque specifications (table query)
print("\n[DEMO 1: Complete Table Retrieval]")
query_with_table_awareness("List all torque specifications from the manual")

# Demo 2: Specific table value
print("\n[DEMO 2: Specific Table Value]")
query_with_table_awareness("What is the wheel nut torque specification?")

# Demo 3: Part numbers table
print("\n[DEMO 3: Part Numbers Table]")
query_with_table_awareness("List all part numbers mentioned in the manual")

# Demo 4: Comparison query requiring table data
print("\n[DEMO 4: Comparison Query]")
query_with_table_awareness("Compare the torque specifications for wheel nut and valve stem")

# ============================================
# INTERACTIVE MODE
# ============================================

print("\n" + "=" * 60)
print("INTERACTIVE TABLE-AWARE MODE")
print("Try queries like:")
print("  - List all torque specifications")
print("  - Show me all part numbers")
print("  - What are the torque values in the manual?")
print("  - Compare wheel nut and axle nut torque")
print("\nType 'quit' to exit")
print("=" * 60)

while True:
    user_input = input("\nYour question: ")
    
    if user_input.lower() in ["quit", "exit", "q"]:
        print("\nSession ended")
        break
    
    if user_input.strip():
        query_with_table_awareness(user_input)
    else:
        print("Please enter a valid question")
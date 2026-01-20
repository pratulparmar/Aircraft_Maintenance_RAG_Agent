# rag_with_citations.py - RAG with Hybrid Search + Smart Citations
import os
import time
from rank_bm25 import BM25Okapi
import numpy as np
import re
from typing import List, Dict, Tuple

print("Starting Advanced RAG System with Citations")
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

# Load PDF
print("\nLoading maintenance manual...")
start = time.time()
loader = PyPDFLoader("test_manual.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} page(s) in {time.time() - start:.1f} seconds")

# Enhanced text splitter with metadata preservation
print("\nCreating text chunks with metadata...")
start = time.time()

class MetadataEnhancedSplitter:
    """Custom splitter that preserves and enhances metadata"""
    
    def __init__(self, chunk_size=800, chunk_overlap=100):
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def extract_section_info(self, text: str) -> Dict:
        """Extract section information from text"""
        metadata = {
            'section_number': None,
            'section_title': None,
            'ata_chapter': None
        }
        
        # Extract ATA Chapter
        ata_match = re.search(r'ATA Chapter (\d+)[:\s-]+([^\n]+)', text)
        if ata_match:
            metadata['ata_chapter'] = f"ATA {ata_match.group(1)}"
            metadata['ata_chapter_name'] = ata_match.group(2).strip()
        
        # Extract Section number and title
        section_match = re.search(r'SECTION (\d+):\s*([^\n]+)', text)
        if section_match:
            metadata['section_number'] = section_match.group(1)
            metadata['section_title'] = section_match.group(2).strip()
        
        # Extract step numbers for procedures
        step_match = re.search(r'Step (\d+):', text)
        if step_match:
            metadata['step_number'] = step_match.group(1)
        
        return metadata
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents and enhance metadata"""
        chunks = self.base_splitter.split_documents(documents)
        
        enhanced_chunks = []
        for chunk in chunks:
            # Preserve original metadata
            enhanced_metadata = chunk.metadata.copy()
            
            # Add extracted metadata
            extracted = self.extract_section_info(chunk.page_content)
            enhanced_metadata.update(extracted)
            
            # Create new document with enhanced metadata
            enhanced_chunk = Document(
                page_content=chunk.page_content,
                metadata=enhanced_metadata
            )
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks

# Use enhanced splitter
splitter = MetadataEnhancedSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks with metadata in {time.time() - start:.1f} seconds")

# Show example metadata
if chunks:
    print("\nExample chunk metadata:")
    print(f"  Section: {chunks[0].metadata.get('section_title', 'N/A')}")
    print(f"  ATA Chapter: {chunks[0].metadata.get('ata_chapter', 'N/A')}")
    print(f"  Page: {chunks[0].metadata.get('page', 'N/A')}")

# Extract text for BM25
chunk_texts = [chunk.page_content for chunk in chunks]

# Load embeddings
print("\nLoading embedding model...")
start = time.time()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print(f"Embedding model ready in {time.time() - start:.1f} seconds")

# Create vector database
print("\nInitializing vector database...")
start = time.time()
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db_citations"
)

if len(vectorstore.get()['ids']) == 0:
    vectorstore.add_documents(chunks)
    print(f"Vector database created in {time.time() - start:.1f} seconds")
else:
    print(f"Vector database loaded in {time.time() - start:.1f} seconds")

# Create BM25 index
print("\nCreating BM25 keyword index...")
start = time.time()
tokenized_chunks = [text.lower().split() for text in chunk_texts]
bm25 = BM25Okapi(tokenized_chunks)
print(f"BM25 index created in {time.time() - start:.1f} seconds")

print("\n" + "=" * 60)
print("CITATION SYSTEM INITIALIZED")
print("=" * 60)

# Citation formatter
class CitationFormatter:
    """Format citations in a professional manner"""
    
    @staticmethod
    def format_source(doc: Document, score: float, index: int) -> str:
        """Format a single source citation"""
        metadata = doc.metadata
        
        parts = []
        
        # Section information
        if metadata.get('section_number'):
            section = f"Section {metadata['section_number']}"
            if metadata.get('section_title'):
                section += f": {metadata['section_title']}"
            parts.append(section)
        elif metadata.get('section_title'):
            parts.append(metadata['section_title'])
        
        # Step information
        if metadata.get('step_number'):
            parts.append(f"Step {metadata['step_number']}")
        
        # Page number
        if metadata.get('page') is not None:
            parts.append(f"Page {int(metadata['page']) + 1}")
        
        # Confidence score
        parts.append(f"Confidence: {score:.2f}")
        
        citation = f"[{index}] " + " | ".join(parts)
        return citation
    
    @staticmethod
    def format_ata_reference(chunks: List[Document]) -> str:
        """Extract and format ATA chapter reference"""
        for chunk in chunks:
            ata = chunk.metadata.get('ata_chapter')
            ata_name = chunk.metadata.get('ata_chapter_name')
            if ata:
                if ata_name:
                    return f"{ata} - {ata_name}"
                return ata
        return "Not specified"

# Hybrid search with citation tracking
def hybrid_search_with_citations(question: str, k: int = 3, alpha: float = 0.5) -> Tuple[List[Document], List[float]]:
    """
    Perform hybrid search and return documents with scores
    """
    # Adjust alpha for technical queries
    if any(char.isdigit() for char in question) or any(c in question for c in ['-', '_']):
        alpha = 0.3  # Favor keyword search
    
    # 1. Semantic search
    semantic_results = vectorstore.similarity_search_with_score(question, k=k*2)
    
    # Normalize scores
    if semantic_results:
        max_semantic = max(score for _, score in semantic_results)
        semantic_dict = {
            i: (doc, 1 - score/max_semantic if max_semantic > 0 else 0)
            for i, (doc, score) in enumerate(semantic_results)
        }
    else:
        semantic_dict = {}
    
    # 2. Keyword search (BM25)
    tokenized_query = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    keyword_dict = {
        i: (chunks[i], bm25_scores[i] / max_bm25)
        for i in range(len(chunks))
    }
    
    # 3. Combine scores
    combined = {}
    all_indices = set(semantic_dict.keys()) | set(keyword_dict.keys())
    
    for idx in all_indices:
        semantic_doc, semantic_score = semantic_dict.get(idx, (None, 0))
        keyword_doc, keyword_score = keyword_dict.get(idx, (None, 0))
        
        doc = semantic_doc or keyword_doc
        combined_score = alpha * semantic_score + (1 - alpha) * keyword_score
        
        combined[idx] = (doc, combined_score)
    
    # Sort and return top k
    sorted_results = sorted(combined.items(), key=lambda x: x[1][1], reverse=True)[:k]
    
    docs = [doc for _, (doc, score) in sorted_results]
    scores = [score for _, (doc, score) in sorted_results]
    
    return docs, scores

# Enhanced query function with citations
def query_with_citations(question: str):
    """
    Query the manual with full citation tracking
    """
    print(f"\nQuestion: {question}")
    
    query_start = time.time()
    
    # Perform hybrid search
    print("Performing hybrid search...")
    search_start = time.time()
    
    retrieved_docs, scores = hybrid_search_with_citations(question, k=3)
    search_time = time.time() - search_start
    
    print(f"Retrieved {len(retrieved_docs)} sources in {search_time:.2f}s")
    
    # Build context
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        context_parts.append(f"[Source {i+1}]\n{doc.page_content}")
    
    context = "\n\n".join(context_parts)
    
    # Create prompt
    prompt = f"""You are an aircraft maintenance assistant. Use the provided sources to answer the question.
If the answer is not in the sources, state that clearly.

Sources:
{context}

Question: {question}

Answer:"""
    
    # Generate response
    print("Generating response with citations...")
    llm_start = time.time()
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400
        )
        
        llm_time = time.time() - llm_start
        answer = response.choices[0].message.content
        
        # Format output with citations
        print(f"\nAnswer:")
        print(answer)
        print(f"\nSources:")
        
        formatter = CitationFormatter()
        for i, (doc, score) in enumerate(zip(retrieved_docs, scores), 1):
            citation = formatter.format_source(doc, score, i)
            print(citation)
        
        # ATA reference
        ata_ref = formatter.format_ata_reference(retrieved_docs)
        print(f"\nReference: {ata_ref}")
        
        # Performance metrics
        total_time = time.time() - query_start
        print(f"\nPerformance: Search={search_time:.2f}s | LLM={llm_time:.2f}s | Total={total_time:.2f}s")
        print("-" * 60)
        
        return answer
        
    except Exception as e:
        print(f"Error: {e}")
        return None

# Demonstration
print("\n" + "=" * 60)
print("TESTING CITATION SYSTEM")
print("=" * 60)

# Test queries
test_queries = [
    "What tools do I need to change a tire?",
    "What is the torque specification for wheel nuts?",
    "What are the safety warnings?",
    "What is the part number for the main wheel assembly?",
    "Describe step 5 of the procedure"
]

for query in test_queries:
    query_with_citations(query)

# Interactive mode
print("\n" + "=" * 60)
print("INTERACTIVE MODE WITH CITATIONS")
print("Type 'quit' to exit")
print("=" * 60)

while True:
    user_input = input("\nYour question: ")
    
    if user_input.lower() in ["quit", "exit", "q"]:
        print("\nSession ended")
        break
    
    if user_input.strip():
        query_with_citations(user_input)
    else:
        print("Please enter a valid question")
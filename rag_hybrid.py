# rag_hybrid.py - RAG with Hybrid Search (Semantic + Keyword)
import os
import time
from rank_bm25 import BM25Okapi
import numpy as np

print("Starting Advanced RAG System with Hybrid Search")
print("=" * 60)

print("\nLoading dependencies...")
start = time.time()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
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

# Create chunks
print("\nCreating text chunks...")
start = time.time()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks in {time.time() - start:.1f} seconds")

# Extract text content for BM25
chunk_texts = [chunk.page_content for chunk in chunks]

# Load embeddings
print("\nLoading embedding model...")
start = time.time()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print(f"Embedding model ready in {time.time() - start:.1f} seconds")

# Create vector database
print("\nInitializing vector database...")
start = time.time()
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db_hybrid")

if len(vectorstore.get()['ids']) == 0:
    vectorstore.add_documents(chunks)
    print(f"Vector database created in {time.time() - start:.1f} seconds")
else:
    print(f"Vector database loaded in {time.time() - start:.1f} seconds")

# Create BM25 index for keyword search
print("\nCreating BM25 keyword index...")
start = time.time()
tokenized_chunks = [text.lower().split() for text in chunk_texts]
bm25 = BM25Okapi(tokenized_chunks)
print(f"BM25 index created in {time.time() - start:.1f} seconds")

print("\n" + "=" * 60)
print("HYBRID SEARCH SYSTEM INITIALIZED")
print("=" * 60)

# Hybrid retrieval function
def hybrid_search(question, k=3, alpha=0.5):
    """
    Perform hybrid search combining:
    - Semantic search (vector similarity)
    - Keyword search (BM25)
    
    Args:
        question: Query string
        k: Number of results to return
        alpha: Weight between semantic (1.0) and keyword (0.0) search
               0.5 = equal weight
    
    Returns:
        List of top k document chunks
    """
    # 1. Semantic search (vector similarity)
    semantic_docs = vectorstore.similarity_search_with_score(question, k=k*2)
    
    # Normalize semantic scores (0-1 range)
    if semantic_docs:
        max_semantic = max(score for _, score in semantic_docs)
        semantic_results = {
            doc.page_content: (1 - score/max_semantic) if max_semantic > 0 else 0
            for doc, score in semantic_docs
        }
    else:
        semantic_results = {}
    
    # 2. Keyword search (BM25)
    tokenized_query = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Normalize BM25 scores (0-1 range)
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    keyword_results = {
        chunk_texts[i]: bm25_scores[i] / max_bm25
        for i in range(len(chunk_texts))
    }
    
    # 3. Combine scores with alpha weighting
    combined_scores = {}
    all_texts = set(semantic_results.keys()) | set(keyword_results.keys())
    
    for text in all_texts:
        semantic_score = semantic_results.get(text, 0)
        keyword_score = keyword_results.get(text, 0)
        combined_scores[text] = alpha * semantic_score + (1 - alpha) * keyword_score
    
    # 4. Sort by combined score and return top k
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    
    return [text for text, score in sorted_results], sorted_results

# Enhanced RAG query with hybrid search
def query_manual_hybrid(question, search_mode="hybrid"):
    """
    Query manual using hybrid search.
    
    Args:
        question: User query
        search_mode: "hybrid", "semantic", or "keyword"
    """
    print(f"\nQuestion: {question}")
    print(f"Search mode: {search_mode}")
    
    query_start = time.time()
    
    # Search with selected mode
    search_start = time.time()
    
    if search_mode == "semantic":
        # Pure vector search
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        print(f"Semantic search: {time.time() - search_start:.2f}s")
        
    elif search_mode == "keyword":
        # Pure BM25 search
        tokenized_query = question.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[-3:][::-1]
        context = "\n\n".join([chunk_texts[i] for i in top_indices])
        print(f"Keyword search: {time.time() - search_start:.2f}s")
        
    else:  # hybrid
        # Hybrid search (default)
        alpha = 0.5  # Equal weight to semantic and keyword
        
        # Adjust alpha based on query characteristics
        if any(char.isdigit() for char in question) or any(c in question for c in ['-', '_']):
            # Query has numbers or special chars (likely part number)
            alpha = 0.3  # Favor keyword search
            print("Detected technical identifier - favoring keyword search")
        
        retrieved_texts, scores = hybrid_search(question, k=3, alpha=alpha)
        context = "\n\n".join(retrieved_texts)
        print(f"Hybrid search (alpha={alpha}): {time.time() - search_start:.2f}s")
        
        # Show score breakdown for top result
        if scores:
            print(f"Top result score: {scores[0][1]:.3f}")
    
    # Generate response
    prompt = f"""You are an aircraft maintenance assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have that information in the manual."

Context from maintenance manual:
{context}

Question: {question}

Answer:"""
    
    print("Generating response...")
    llm_start = time.time()
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300
        )
        
        llm_time = time.time() - llm_start
        total_time = time.time() - query_start
        
        answer = response.choices[0].message.content
        print(f"Response time: {llm_time:.2f}s")
        
        print(f"\nAnswer:\n{answer}")
        print(f"\nTotal time: {total_time:.2f}s")
        print("-" * 60)
        
        return answer
        
    except Exception as e:
        print(f"Error: {e}")
        return None

# Demonstration of search modes
print("\n" + "=" * 60)
print("TESTING HYBRID SEARCH CAPABILITIES")
print("=" * 60)

# Test 1: Part number query (should favor keyword search)
print("\n[TEST 1: Part Number Query]")
print("This query contains an exact part number - hybrid search should favor keyword matching")
query_manual_hybrid("What is part number NAS1149F0363P?", "hybrid")

# Test 2: Semantic query (concept-based)
print("\n[TEST 2: Semantic Query]")
print("This query is conceptual - hybrid search will balance both approaches")
query_manual_hybrid("What safety precautions should I take?", "hybrid")

# Test 3: Comparison of search modes
print("\n[TEST 3: Search Mode Comparison]")
test_query = "Find NAS1149F0363P"

print("\n--- Pure Semantic Search ---")
query_manual_hybrid(test_query, "semantic")

print("\n--- Pure Keyword Search ---")
query_manual_hybrid(test_query, "keyword")

print("\n--- Hybrid Search (Best of Both) ---")
query_manual_hybrid(test_query, "hybrid")

# Interactive mode
print("\n" + "=" * 60)
print("INTERACTIVE HYBRID SEARCH")
print("Enter questions to test hybrid retrieval")
print("Type 'quit' to exit")
print("=" * 60)

while True:
    user_input = input("\nYour question: ")
    
    if user_input.lower() in ["quit", "exit", "q"]:
        print("\nSession ended")
        break
    
    if user_input.strip():
        query_manual_hybrid(user_input, "hybrid")
    else:
        print("Please enter a valid question")
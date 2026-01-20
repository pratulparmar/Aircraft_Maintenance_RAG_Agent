# rag_with_groq_timed.py - Aircraft Maintenance RAG System
import os
import time

print("Starting Airbus RAG Agent with Groq...")
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

# Configuration - Add your Groq API key here
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment. Create .env file with your key.")

groq_client = Groq(api_key=GROQ_API_KEY)
print("Groq API connected")

# Load PDF document
print("\nReading maintenance manual...")
start = time.time()
loader = PyPDFLoader("test_manual.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} page(s) in {time.time() - start:.1f} seconds")

# Split document into chunks for better retrieval
print("\nCreating text chunks...")
start = time.time()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks in {time.time() - start:.1f} seconds")

# Initialize embedding model
print("\nLoading embedding model...")
print("Note: First-time load takes 30-60 seconds (downloads model weights)")
print("Subsequent runs will be much faster")
start = time.time()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print(f"Embedding model ready in {time.time() - start:.1f} seconds")

# Create or load vector database
print("\nInitializing vector database...")
start = time.time()

vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db_groq")

if len(vectorstore.get()['ids']) == 0:
    vectorstore.add_documents(chunks)
    print(f"Database created in {time.time() - start:.1f} seconds")
else:
    print(f"Database loaded from disk in {time.time() - start:.1f} seconds")

# Main RAG query function
def ask_rag(question):
    """
    Performs retrieval-augmented generation to answer questions
    using the maintenance manual as context.
    """
    print(f"\nQuestion: {question}")
    
    total_start = time.time()
    
    # Step 1: Retrieve relevant document chunks
    print("Searching database...")
    search_start = time.time()
    relevant_docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    print(f"   Search completed in {time.time() - search_start:.2f} seconds")
    
    # Step 2: Construct prompt with context
    prompt = f"""You are an aircraft maintenance assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, state that the information is not available in the manual.

Context from maintenance manual:
{context}

Question: {question}

Answer:"""
    
    # Step 3: Generate response using Groq LLM
    print("Generating response...")
    groq_start = time.time()
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300
        )
        
        print(f"   Response generated in {time.time() - groq_start:.2f} seconds")
        
        answer = response.choices[0].message.content
        print(f"\nAnswer:\n{answer}")
        print(f"\nTotal query time: {time.time() - total_start:.2f} seconds")
        print("-" * 60)
        return answer
        
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

# System ready notification
print("\n" + "=" * 60)
print("RAG SYSTEM INITIALIZED")
print("=" * 60)

print("\nRunning test queries...\n")

# Test queries
ask_rag("What tools do I need to change a tire?")
ask_rag("What is the torque specification for wheel nuts?")
ask_rag("What are the safety warnings?")
ask_rag("What is the part number for the main wheel assembly?")

# Interactive query mode
print("\n" + "=" * 60)
print("Interactive mode - Enter your questions below")
print("Type 'quit' to exit")
print("=" * 60)

while True:
    user_question = input("\nYour question: ")
    
    if user_question.lower() in ["quit", "exit", "q"]:
        print("\nSession ended")
        break
    
    if user_question.strip():
        ask_rag(user_question)
    else:
        print("Please enter a valid question")

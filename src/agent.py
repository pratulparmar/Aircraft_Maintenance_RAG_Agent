# rag_agent.py - Complete Agentic RAG System
import os
import time
import numpy as np
import re
import json
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Optional

print("Starting Agentic RAG System")
print("=" * 60)

print("\nLoading dependencies...")
start = time.time()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from groq import Groq

print(f"Dependencies loaded in {time.time() - start:.1f} seconds")

# ============================================
# CONFIGURATION - ADD YOUR API KEY HERE
# ============================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment. Create .env file with your key.")


groq_client = Groq(api_key=GROQ_API_KEY)
print("Groq API connected")

# ============================================
# LOAD AND PREPARE DATA
# ============================================

print("\nLoading maintenance manual...")
start = time.time()
loader = PyPDFLoader("test_manual.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} page(s) in {time.time() - start:.1f} seconds")

print("\nCreating text chunks...")
start = time.time()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)
chunk_texts = [chunk.page_content for chunk in chunks]
print(f"Created {len(chunks)} chunks in {time.time() - start:.1f} seconds")

print("\nLoading embedding model...")
start = time.time()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print(f"Embedding model ready in {time.time() - start:.1f} seconds")

print("\nInitializing vector database...")
start = time.time()
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db_agent"
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
print("BASE SYSTEM INITIALIZED")
print("=" * 60)

# ============================================
# ENGINEERING TOOLS
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
        "psi_to_bar": 0.0689476,
        "bar_to_psi": 14.5038,
        "in_to_mm": 25.4,
        "mm_to_in": 0.0393701,
        "in_to_cm": 2.54,
        "cm_to_in": 0.393701,
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
        else:
            return {
                "error": f"Conversion from {from_unit} to {to_unit} not supported"
            }

class ChecklistGenerator:
    """Generate maintenance checklists in JSON format"""
    
    @staticmethod
    def generate_checklist(
        procedure_name: str,
        tools: List[str],
        steps: List[str],
        safety_warnings: List[str],
        ata_chapter: Optional[str] = None
    ) -> Dict[str, Any]:
        
        checklist = {
            "procedure": procedure_name,
            "ata_chapter": ata_chapter,
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
                    "completed": False,
                    "verified_by": None
                }
                for i, step in enumerate(steps)
            ],
            "post_work_checks": {
                "all_steps_completed": False,
                "tools_returned": False,
                "documentation_complete": False
            },
            "sign_off": {
                "technician": None,
                "supervisor": None,
                "date": None
            }
        }
        
        return checklist

class SafetyChecker:
    """Check procedures against safety compliance rules"""
    
    MANDATORY_CHECKS = {
        "parking_brake": ["parking brake", "brake engaged"],
        "safety_chocks": ["safety chocks", "chocks"],
        "level_ground": ["level ground", "level surface"],
        "hydraulic_pressure": ["hydraulic pressure", "pressure zero"]
    }
    
    @staticmethod
    def check_compliance(procedure_text: str) -> Dict[str, Any]:
        text_lower = procedure_text.lower()
        
        compliance = {
            "compliant": True,
            "checks_passed": [],
            "checks_failed": [],
            "warnings": []
        }
        
        for check_name, keywords in SafetyChecker.MANDATORY_CHECKS.items():
            found = any(keyword in text_lower for keyword in keywords)
            
            if found:
                compliance["checks_passed"].append(check_name)
            else:
                compliance["checks_failed"].append(check_name)
                compliance["compliant"] = False
                compliance["warnings"].append(
                    f"SAFETY WARNING: {check_name.replace('_', ' ').title()} not mentioned"
                )
        
        return compliance

# ============================================
# RETRIEVAL FUNCTION
# ============================================

def retrieve_context(question: str, k: int = 3) -> tuple:
    """Retrieve relevant context using hybrid search"""
    
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
        
        combined_score = alpha * semantic_score + (1 - alpha) * keyword_score
        combined[idx] = (chunks[idx], combined_score)
    
    sorted_results = sorted(combined.items(), key=lambda x: x[1][1], reverse=True)[:k]
    
    docs = [doc for _, (doc, score) in sorted_results]
    context = "\n\n".join([doc.page_content for doc in docs])
    
    return context, docs

# ============================================
# AGENT SYSTEM
# ============================================

class MaintenanceAgent:
    """Intelligent agent that can use tools"""
    
    def __init__(self, groq_client):
        self.groq_client = groq_client
        self.unit_converter = UnitConverter()
        self.checklist_generator = ChecklistGenerator()
        self.safety_checker = SafetyChecker()
    
    def analyze_query(self, question: str) -> Dict[str, Any]:
        """Analyze query to determine needed tools"""
        analysis = {
            "needs_retrieval": True,
            "needs_unit_conversion": False,
            "needs_checklist": False,
            "needs_safety_check": False,
            "query_type": "informational"
        }
        
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["convert", "n-m", "nm", "newton", "metric", "kpa"]):
            analysis["needs_unit_conversion"] = True
        
        if any(word in question_lower for word in ["checklist", "procedure", "steps", "replace", "install", "change"]):
            analysis["needs_checklist"] = True
            analysis["query_type"] = "procedural"
        
        if any(word in question_lower for word in ["safe", "safety", "warning", "danger", "caution"]):
            analysis["needs_safety_check"] = True
        
        return analysis
    
    def process_query(self, question: str) -> Dict[str, Any]:
        """Process a query using available tools"""
        print(f"\nAgent analyzing query: {question}")
        query_start = time.time()
        
        analysis = self.analyze_query(question)
        print(f"Query type: {analysis['query_type']}")
        
        if analysis['needs_checklist']:
            print("Tool selection: Checklist generation required")
        if analysis['needs_unit_conversion']:
            print("Tool selection: Unit conversion required")
        if analysis['needs_safety_check']:
            print("Tool selection: Safety compliance check required")
        
        print("\nRetrieving relevant context...")
        context, docs = retrieve_context(question)
        
        prompt = self._build_agent_prompt(question, context, analysis)
        
        print("Agent reasoning and generating response...")
        llm_start = time.time()
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=800
            )
            
            llm_time = time.time() - llm_start
            raw_answer = response.choices[0].message.content
            
            if not raw_answer or len(raw_answer.strip()) == 0:
                return {
                    "error": "Empty response from LLM",
                    "performance": {"total_time": time.time() - query_start}
                }
            
            result = self._apply_tools(raw_answer, context, analysis)
            result["performance"] = {
                "total_time": time.time() - query_start,
                "llm_time": llm_time
            }
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "performance": {"total_time": time.time() - query_start}
            }
    
    def _build_agent_prompt(self, question: str, context: str, analysis: Dict) -> str:
        """Build prompt based on query analysis"""
        
        base_prompt = f"""You are an expert aircraft maintenance assistant.

Context from manual:
{context}

Question: {question}

"""
        
        if analysis['needs_checklist']:
            base_prompt += """
Provide a detailed answer AND extract the following for checklist generation:
- Procedure name
- Required tools (list each)
- Procedure steps (list each)
- Safety warnings

Format:
ANSWER: [Your answer]

CHECKLIST_DATA:
PROCEDURE: [name]
TOOLS: [tool1], [tool2], [tool3]
STEPS: [step1] | [step2] | [step3]
SAFETY: [warning1] | [warning2]
"""
        else:
            base_prompt += "\nProvide a clear, concise answer based on the context."
        
        return base_prompt
    
    def _apply_tools(self, llm_response: str, context: str, analysis: Dict) -> Dict[str, Any]:
        """Apply tools based on analysis"""
        
        result = {
            "answer": llm_response,
            "tools_used": []
        }
        
        if "CHECKLIST_DATA:" in llm_response:
            parts = llm_response.split("CHECKLIST_DATA:")
            result["answer"] = parts[0].replace("ANSWER:", "").strip()
            checklist_data = parts[1]
            
            try:
                procedure = re.search(r'PROCEDURE:\s*(.+?)(?:\n|$)', checklist_data)
                tools = re.search(r'TOOLS:\s*(.+?)(?:\n|$)', checklist_data)
                steps = re.search(r'STEPS:\s*(.+?)(?:\n|$)', checklist_data)
                safety = re.search(r'SAFETY:\s*(.+?)(?:\n|$)', checklist_data)
                
                if all([procedure, tools, steps, safety]):
                    checklist = self.checklist_generator.generate_checklist(
                        procedure_name=procedure.group(1).strip(),
                        tools=[t.strip() for t in tools.group(1).split(',')],
                        steps=[s.strip() for s in steps.group(1).split('|')],
                        safety_warnings=[w.strip() for w in safety.group(1).split('|')],
                        ata_chapter="ATA 32 - Landing Gear"
                    )
                    
                    result["checklist"] = checklist
                    result["tools_used"].append("checklist_generator")
            except Exception as e:
                result["checklist_error"] = str(e)
        
        if analysis['needs_safety_check']:
            compliance = self.safety_checker.check_compliance(context)
            result["safety_compliance"] = compliance
            result["tools_used"].append("safety_checker")
        
        if analysis['needs_unit_conversion']:
            torque_matches = re.findall(r'(\d+)\s*(ft-lbs|in-lbs)', result["answer"])
            if torque_matches:
                conversions = []
                for value, unit in torque_matches:
                    if unit == "ft-lbs":
                        conv = self.unit_converter.convert(float(value), "ft-lbs", "nm")
                        conversions.append(conv)
                    elif unit == "in-lbs":
                        conv = self.unit_converter.convert(float(value), "in-lbs", "nm")
                        conversions.append(conv)
                
                if conversions:
                    result["unit_conversions"] = conversions
                    result["tools_used"].append("unit_converter")
        
        return result
    
    def format_output(self, result: Dict[str, Any]) -> str:
        """Format agent output for display"""
        
        output = []
        output.append("=" * 60)
        output.append("AGENT RESPONSE")
        output.append("=" * 60)
        
        if "error" in result:
            output.append(f"\nError: {result['error']}")
            if "performance" in result:
                output.append(f"Time: {result['performance']['total_time']:.2f}s")
            output.append("=" * 60)
            return "\n".join(output)
        
        if "answer" in result and result["answer"]:
            output.append(f"\n{result['answer']}")
        else:
            output.append("\nNo answer generated")
        
        if result.get('tools_used'):
            output.append(f"\nTools used: {', '.join(result['tools_used'])}")
        
        if result.get('unit_conversions'):
            output.append("\nUnit Conversions:")
            for conv in result['unit_conversions']:
                if 'formatted' in conv:
                    output.append(f"  - {conv['formatted']}")
        
        if result.get('safety_compliance'):
            comp = result['safety_compliance']
            output.append("\nSafety Compliance Check:")
            output.append(f"  Status: {'PASS' if comp['compliant'] else 'FAIL'}")
            output.append(f"  Checks passed: {len(comp['checks_passed'])}")
            if comp['warnings']:
                output.append("  Warnings:")
                for warning in comp['warnings']:
                    output.append(f"    - {warning}")
        
        if result.get('checklist'):
            output.append("\nGenerated Checklist (JSON):")
            output.append(json.dumps(result['checklist'], indent=2))
        
        if result.get('performance'):
            perf = result['performance']
            output.append(f"\nPerformance: Total={perf['total_time']:.2f}s")
        
        output.append("=" * 60)
        
        return "\n".join(output)

# ============================================
# INITIALIZE AGENT
# ============================================

print("\nInitializing agent with tools...")
agent = MaintenanceAgent(groq_client)

print("\n" + "=" * 60)
print("AGENTIC SYSTEM READY")
print("=" * 60)
print("\nAvailable capabilities:")
print("  - Hybrid retrieval (semantic + keyword)")
print("  - Unit conversion (ft-lbs, psi, inches)")
print("  - Checklist generation (JSON format)")
print("  - Safety compliance checking")
print("  - Multi-step reasoning")
print("=" * 60)

# ============================================
# DEMONSTRATIONS
# ============================================

print("\n" + "=" * 60)
print("AGENT DEMONSTRATIONS")
print("=" * 60)

print("\n[DEMO 1: Simple Information Retrieval]")
result = agent.process_query("What tools do I need to change a tire?")
print(agent.format_output(result))

print("\n[DEMO 2: Query with Unit Conversion]")
result = agent.process_query("What is the wheel nut torque? Also convert to Newton-meters")
print(agent.format_output(result))

print("\n[DEMO 3: Checklist Generation]")
result = agent.process_query("I need to change the aircraft tire. Generate a checklist for me.")
print(agent.format_output(result))

print("\n[DEMO 4: Safety Compliance Check]")
result = agent.process_query("What safety precautions should I take for this procedure?")
print(agent.format_output(result))

# ============================================
# INTERACTIVE MODE
# ============================================

print("\n" + "=" * 60)
print("INTERACTIVE AGENTIC MODE")
print("Try commands like:")
print("  - Generate a checklist for tire replacement")
print("  - Convert 450 ft-lbs to Newton-meters")
print("  - What are the safety requirements?")
print("  - List all torque specifications in metric units")
print("\nType 'quit' to exit")
print("=" * 60)

while True:
    user_input = input("\nYour question: ")
    
    if user_input.lower() in ["quit", "exit", "q"]:
        print("\nSession ended")
        break
    
    if user_input.strip():
        result = agent.process_query(user_input)
        print(agent.format_output(result))
    else:
        print("Please enter a valid question")
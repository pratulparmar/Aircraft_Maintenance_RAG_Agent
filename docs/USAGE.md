# Usage Guide

## Quick Start

1. Start the application:
```bash
streamlit run app.py
```

2. Upload a PDF manual using the sidebar
3. Click "Process PDF"
4. Enter your query or use quick action buttons

## Query Examples

### Basic Information
```
What tools do I need to change a tire?
List all part numbers in the manual
What are the torque specifications?
```

### Checklist Generation
```
Generate a checklist for tire replacement
Create a maintenance procedure checklist
I need to replace the fuel pump, generate checklist
```

### Unit Conversion
```
Convert 450 ft-lbs to Newton-meters
What is the wheel nut torque in metric units?
Show me all torque specs in N-m
```

### Safety Queries
```
What are the safety warnings?
What safety precautions should I take?
Check safety compliance for this procedure
```

### Technical Queries
```
Find part number NAS1149F0363P
What is the specification for MS28889-2?
Compare wheel nut and valve stem torque
```

## Understanding Results

### Answer Section
Main response to your query with relevant information extracted from the manual.

### Tools Used
Shows which engineering tools were applied:
- `checklist_generator` - JSON checklist created
- `unit_converter` - Units converted
- `safety_checker` - Safety compliance verified

### Sources
Expandable sections showing:
- Retrieved document chunks
- Confidence scores
- Page numbers

### Performance Metrics
- Retrieval time: How long to search the manual
- LLM time: How long to generate the answer
- Total time: Complete query processing time

## Query Modes

### Standard Q&A
Direct question answering with source citations.

### Generate Checklist
Creates structured JSON maintenance checklist including:
- Required tools
- Procedure steps
- Safety warnings
- Sign-off tracking

Download using the "Download Checklist (JSON)" button.

### Unit Conversion
Automatically detects and converts engineering units:
- Torque: ft-lbs ↔ N-m, in-lbs ↔ N-m
- Pressure: PSI ↔ kPa
- Length: inches ↔ mm

## Advanced Features

### Query History
View recent queries in the expandable history section.

### Source Citations
Enable "Show source citations" to see:
- Which document sections were used
- Confidence scores for retrieval
- Page references

### Performance Metrics
Enable "Show performance metrics" to track:
- System response times
- Component performance breakdown

## Command Line Usage

For terminal-based interaction:
```bash
python rag_agent.py
```

This provides:
- Interactive query mode
- Direct tool access
- Performance benchmarking
- Debugging output

## Integration

### JSON Output

Checklists are generated in JSON format for easy integration:
```json
{
  "procedure": "Aircraft Tire Change",
  "required_tools": [...],
  "procedure_steps": [...],
  "sign_off": {...}
}
```

Use this format for:
- Maintenance execution systems
- Work order generation
- Compliance tracking
- Digital workflows

## Best Practices

1. **Be specific**: "What is the wheel nut torque?" is better than "Tell me about torque"

2. **Use technical terms**: The system recognizes part numbers, ATA chapters, and technical terminology

3. **Request structure when needed**: Ask for checklists or tables explicitly

4. **Verify critical information**: Always verify torque specs and safety procedures against the original manual

5. **Save important outputs**: Download generated checklists for record-keeping

## Tips

- Use quick action buttons for common queries
- Part numbers are detected automatically
- The system maintains context within a session
- Clear technical queries get better results than vague ones
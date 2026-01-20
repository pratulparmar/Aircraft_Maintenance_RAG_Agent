import os
from pathlib import Path

def display_tree(directory, prefix="", ignore_dirs=None):
    if ignore_dirs is None:
        ignore_dirs = {'venv', '__pycache__', '.git', 'chroma_db_groq', 
                       'chroma_db', 'chroma_db_agent', 'chroma_db_hybrid',
                       'chroma_db_citations', 'chroma_db_tables', 'chroma_streamlit',
                       '.vscode'}
    
    directory = Path(directory)
    
    try:
        items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
    except PermissionError:
        return
    
    items = [item for item in items if item.name not in ignore_dirs]
    
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{item.name}")
        
        if item.is_dir():
            extension = "    " if is_last else "│   "
            display_tree(item, prefix + extension, ignore_dirs)

print("\nCurrent Project Structure:")
print("=" * 60)
print("Airbus_rag_agent/")
display_tree(".")
print("=" * 60)
from llama_index.core import SimpleDirectoryReader
import os

pdf_path = r"d:\AI_Agent\CS311.pdf"

if not os.path.exists(pdf_path):
    print(f"File not found: {pdf_path}")
else:
    try:
        reader = SimpleDirectoryReader(input_files=[pdf_path])
        documents = reader.load_data()
        
        print(f"Loaded {len(documents)} pages/documents.")
        for i, doc in enumerate(documents):
            print(f"--- Page {i+1} ---")
            print(doc.text)
            print("\n")
    except Exception as e:
        print(f"Error reading PDF: {e}")

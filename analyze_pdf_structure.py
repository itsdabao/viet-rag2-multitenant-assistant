
from llama_index.core import SimpleDirectoryReader
import re
import sys

def analyze_pdf(path):
    try:
        print(f"Loading {path} with SimpleDirectoryReader...")
        reader = SimpleDirectoryReader(input_files=[path])
        docs = reader.load_data()
        
        full_text = ""
        for d in docs:
            full_text += d.text + "\n"
            
        print(f"Total Text Length: {len(full_text)}")
        
        # Normalize newlines
        lines = full_text.split('\n')
        
        chapter_content = {}
        current_chapter = None
        
        # Regex for headers
        patterns = [
            r"^\s*(Chương|Chapter)\s+2\b",
            r"^\s*(Chương|Chapter)\s+3\b",
            r"^\s*(Chương|Chapter)\s+4\b" 
        ]
        
        print("Scanning for chapters...")
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line: continue
            
            if re.match(patterns[0], line, re.IGNORECASE):
                print(f"Found Chapter 2 at line {i}")
                current_chapter = 2
                chapter_content[2] = []
            elif re.match(patterns[1], line, re.IGNORECASE):
                print(f"Found Chapter 3 at line {i}")
                current_chapter = 3
                chapter_content[3] = []
            elif re.match(patterns[2], line, re.IGNORECASE):
                print(f"Found Chapter 4 at line {i} (Stopping capture)")
                current_chapter = 4
                
            if current_chapter in (2, 3):
                chapter_content[current_chapter].append(line)
                
        if 2 in chapter_content and 3 in chapter_content:
            c2_text = "\n".join(chapter_content[2])
            c3_text = "\n".join(chapter_content[3])
            
            print(f"Chapter 2 Length: {len(c2_text)} chars")
            print(f"Chapter 3 Length: {len(c3_text)} chars")
            
            words2 = set(c2_text.lower().split())
            words3 = set(c3_text.lower().split())
            
            intersection = len(words2.intersection(words3))
            union = len(words2.union(words3))
            jaccard = intersection / union if union else 0
            
            print(f"Jaccard Similarity: {jaccard:.4f}")
            
            print("Chapter 2 First 300 chars:")
            print(c2_text[:300])
            print("---")
            print("Chapter 3 First 300 chars:")
            print(c3_text[:300])
            
        else:
            print("Could not find both chapters.")
            print(f"Chapters found: {list(chapter_content.keys())}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # analyze_pdf("d:\\AI_Agent\\CS311_new.pdf")
    # Simplify to just print text for evaluation
    try:
        print("Reading CS311.pdf...")
        reader = SimpleDirectoryReader(input_files=[r"d:\AI_Agent\CS311.pdf"])
        docs = reader.load_data()
        for doc in docs:
            print(doc.text)
            print("-" * 20)
    except Exception as e:
        print(f"Error: {e}")

"""
Comprehensive Chunking Test Suite
So sánh Fixed-Size vs Document-Based (Structure) Chunking
"""

from pathlib import Path
import sys

# Fix encoding cho Windows
import codecs
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add project root to path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from src.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from src.data_loader import load_documents
from src.chunking_strategies import TextNormalizer, DocumentBasedParser, get_node_parser
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)


def print_section(title):
    """Print formatted section."""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)


def display_all_chunks(nodes, title):
    """Hiển thị toàn bộ chunks."""
    print(f"\n[{title}] Total: {len(nodes)} chunks\n")
    
    for i, node in enumerate(nodes, 1):
        text = node.get_content()
        first_line = text.split('\n')[0][:70] if '\n' in text else text[:70]
        
        print(f"  [{i:2d}] ({len(text):4d} chars) {first_line}...")
    
    print()


def test_chunking_comparison():
    """
    So sánh Fixed-Size vs Document-Based Chunking
    """
    
    print_header("CHUNKING COMPARISON TEST")
    print("Fixed-Size Chunking vs Document-Based (Structure) Chunking")
    
    # ==================== STEP 1: Select PDF ====================
    print_section("STEP 1: Select PDF File")
    
    pdf_files = sorted(Path(DATA_PATH).glob("*.pdf"))
    
    if not pdf_files:
        print("\n[X] Khong tim thay file PDF nao!")
        return None
    
    print(f"\n[*] Tim thay {len(pdf_files)} file PDF:")
    for i, pdf in enumerate(pdf_files, 1):
        size_kb = pdf.stat().st_size / 1024
        print(f"  [{i}] {pdf.name} ({size_kb:.1f} KB)")
    
    print("\nNhap so thu tu file (Enter = 1):")
    choice = input(">>> ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(pdf_files):
        selected_pdf = pdf_files[int(choice) - 1]
    else:
        selected_pdf = pdf_files[0]
    
    print(f"\n[OK] Chon: {selected_pdf.name}")
    
    # ==================== STEP 2: Load PDF ====================
    print_section("STEP 2: Load PDF")
    
    try:
        documents = load_documents(str(selected_pdf.parent), input_files=[str(selected_pdf)])
        if not documents:
            print("[X] Khong the load!")
            return None
        
        # Merge all pages
        print(f"[*] PDF co {len(documents)} pages")
        all_texts = [doc.get_content() for doc in documents]
        original_text = "\n\n".join(all_texts)
        
        doc = Document(text=original_text)
        print(f"[OK] Total: {len(original_text):,} chars")
        
    except Exception as e:
        print(f"[X] Error: {e}")
        return None
    
    # ==================== STEP 3: Normalize ====================
    print_section("STEP 3: Text Normalization")
    
    normalizer = TextNormalizer()
    normalized_text = normalizer.normalize(original_text)
    normalized_doc = Document(text=normalized_text)
    
    level1 = normalized_text.count('\n## ')
    level2 = normalized_text.count('\n### ')
    print(f"[OK] Headers detected: {level1} (##) + {level2} (###) = {level1+level2} total")
    
    # Hiển thị toàn bộ văn bản đã normalize
    print(f"\n[NORMALIZED TEXT] ({len(normalized_text):,} chars)")
    print("-" * 80)
    print(normalized_text)
    print("-" * 80)    
    # ==================== STEP 4: Fixed-Size Chunking ====================
    print_section("STEP 4: FIXED-SIZE CHUNKING")
    print(f"Settings: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    
    fixed_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    fixed_nodes = fixed_parser.get_nodes_from_documents([doc])
    
    display_all_chunks(fixed_nodes, "FIXED-SIZE CHUNKS")
    
    # ==================== STEP 5: Document-Based Chunking ====================
    print_section("STEP 5: DOCUMENT-BASED (STRUCTURE) CHUNKING")
    print("Settings: auto_normalize=True, structure-based ##/###")
    
    doc_parser = DocumentBasedParser(auto_normalize=False)  # Already normalized
    doc_nodes = doc_parser.get_nodes_from_documents([normalized_doc])
    
    display_all_chunks(doc_nodes, "DOCUMENT-BASED CHUNKS")
    
    # ==================== STEP 6: Simple Comparison ====================
    print_header("COMPARISON SUMMARY")
    
    fixed_sizes = [len(n.get_content()) for n in fixed_nodes]
    doc_sizes = [len(n.get_content()) for n in doc_nodes]
    
    print(f"""
+------------------------------+----------------+----------------+
| Metric                       | Fixed-Size     | Document-Based |
+------------------------------+----------------+----------------+
| Total Chunks                 | {len(fixed_nodes):>14} | {len(doc_nodes):>14} |
| Avg Size (chars)             | {sum(fixed_sizes)//len(fixed_sizes) if fixed_sizes else 0:>14} | {sum(doc_sizes)//len(doc_sizes) if doc_sizes else 0:>14} |
| Min Size                     | {min(fixed_sizes) if fixed_sizes else 0:>14} | {min(doc_sizes) if doc_sizes else 0:>14} |
| Max Size                     | {max(fixed_sizes) if fixed_sizes else 0:>14} | {max(doc_sizes) if doc_sizes else 0:>14} |
+------------------------------+----------------+----------------+
""")
    
    # ==================== STEP 7: Sample Chunks ====================
    print_section("SAMPLE: First Chunk from Each Method")
    
    print("\n[FIXED-SIZE - Chunk 1]")
    print("-" * 80)
    print(fixed_nodes[0].get_content()[:600] if fixed_nodes else "N/A")
    print("-" * 80)
    
    print("\n[DOCUMENT-BASED - Chunk 1]")
    print("-" * 80)
    print(doc_nodes[0].get_content()[:600] if doc_nodes else "N/A")
    print("-" * 80)
    
    print_header("TEST COMPLETE")
    
    return {
        'fixed_nodes': fixed_nodes,
        'doc_nodes': doc_nodes
    }


def quick_test():
    """Quick test with sample text."""
    
    print_header("QUICK TEST - SAMPLE TEXT")
    
    sample = """
1. Gioi thieu

BrightPath Academy la trung tam Anh ngu hang dau. Chung toi chuyen cung cap cac khoa hoc chat luong cao.

1.1. Ve trung tam

Thanh lap nam 2010, hien co 5 co so tai TP.HCM. Doi ngu giao vien ban ngu va Viet Nam giau kinh nghiem.

1.2. Su menh

Giup hoc vien tu tin giao tiep tieng Anh trong moi tinh huong.

2. Cac khoa hoc

Chung toi co nhieu khoa hoc phu hop cho moi luu tuong va nhu cau.

2.1. Tieng Anh giao tiep

Khoa hoc danh cho nguoi muon nang cao ky nang giao tiep. Thoi luong 3 thang, hoc phi 3.500.000 VND.

2.2. Luyen thi IELTS

Khoa hoc luyen thi IELTS muc tieu 6.5+. Thoi luong 6 thang, hoc phi 12.000.000 VND.

3. Hoc phi va chinh sach

3.1. Bang gia

- Giao tiep: 3.500.000 VND
- IELTS: 12.000.000 VND

3.2. Uu dai

- Dong 1 lan giam 10%
- Gioi thieu ban be giam 5%
"""
    
    # Normalize
    normalizer = TextNormalizer()
    normalized = normalizer.normalize(sample)
    
    doc = Document(text=sample)
    norm_doc = Document(text=normalized)
    
    # Fixed-size
    print_section("FIXED-SIZE CHUNKING (chunk_size=800)")
    fixed_parser = SentenceSplitter(chunk_size=800, chunk_overlap=100)
    fixed_nodes = fixed_parser.get_nodes_from_documents([doc])
    display_all_chunks(fixed_nodes, "FIXED-SIZE")
    
    # Document-based
    print_section("DOCUMENT-BASED CHUNKING (structure ##/###)")
    doc_parser = DocumentBasedParser(auto_normalize=False)
    doc_nodes = doc_parser.get_nodes_from_documents([norm_doc])
    display_all_chunks(doc_nodes, "DOCUMENT-BASED")
    
    # Summary
    print_header("SUMMARY")
    print(f"Fixed-Size: {len(fixed_nodes)} chunks")
    print(f"Document-Based: {len(doc_nodes)} chunks")
    
    print("\n[Document-Based Chunks Detail]")
    for i, node in enumerate(doc_nodes, 1):
        text = node.get_content()
        print(f"\n--- Chunk {i} ---")
        print(text[:300])
        if len(text) > 300:
            print("...")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CHUNKING COMPARISON: Fixed-Size vs Document-Based".center(80))
    print("=" * 80)
    
    print("\nChon mode:")
    print("[1] Quick Test - Sample Text")
    print("[2] Full Test - Real PDF")
    
    choice = input("\nLua chon (Enter = 2): ").strip() or "2"
    
    if choice == "1":
        quick_test()
    elif choice == "2":
        test_chunking_comparison()
    else:
        print("Invalid choice!")

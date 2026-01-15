# Structure-Based Chunking với TextNormalizer

## 📚 Tổng Quan

Hệ thống chunking nâng cao với 3 thành phần chính:

1. **TextNormalizer**: Tự động phát hiện và thêm Markdown headers (##/###)
2. **DocumentBasedParser**: Chia theo cấu trúc Markdown
3. **Context Injection**: Subsections tự động có header cha

## 🎯 Tính Năng

### 1. TextNormalizer - Auto-Detect Headers

Tự động chuyển đổi văn bản thô thành Markdown:

**Level 1 Headers (##):**
- `1. Giới thiệu` → `## 1. Giới thiệu`
- `THÔNG TIN CHUNG` → `## THÔNG TIN CHUNG`

**Level 2 Headers (###):**
- `1.1. Học phí cơ bản` → `### 1.1. Học phí cơ bản`
- `5.2. Giáo viên` → `### 5.2. Giáo viên`

**Thông minh:**
- ✅ Chỉ xử lý dòng ở đầu văn bản (`^`)
- ✅ Kiểm tra độ dài và chữ hoa để tránh nhầm với list items
- ✅ Không xử lý lại nếu đã có Markdown

### 2. Structure-Based Chunking

**Nguyên tắc cắt:**
- Gặp `## ` (Section) → **Bắt buộc cắt chunk mới**
- Gặp `### ` (Sub-section) → Tách riêng nếu đủ dài
- Mỗi chunk là 1 đơn vị ý nghĩa hoàn chỉnh

**Context Injection:**
```
Văn bản gốc:
## 2. Học phí
### 2.1. Chính sách giảm giá
Nội dung...

Chunk được tạo:
## 2. Học phí
### 2.1. Chính sách giảm giá
Nội dung...
```
→ AI biết "Chính sách giảm giá" thuộc về "Học phí"

### 3. Fallback Mechanisms

- Không có cấu trúc Markdown → Chia theo paragraph
- Chunk quá lớn → Chia theo câu
- Câu quá dài → Chia cứng theo max_chunk_size

## 🚀 Cách Sử Dụng

### 1. Cấu hình (src/config.py)

```python
# Bật structure-based chunking
CHUNKING_STRATEGY = "document_based"  

# Tham số
DOC_BASED_MIN_CHUNK_SIZE = 200
DOC_BASED_MAX_CHUNK_SIZE = 1500
```

### 2. Sử dụng trong Code

```python
from src.chunking_strategies import DocumentBasedParser, TextNormalizer

# Option 1: Auto-normalize (khuyến nghị)
parser = DocumentBasedParser(auto_normalize=True)
nodes = parser.get_nodes_from_documents([doc])

# Option 2: Manual normalize
normalizer = TextNormalizer()
normalized_text = normalizer.normalize(raw_text)
# ... then parse
```

### 3. Chạy Ingest

```bash
# Ingest với structure-based chunking
python ingest.py --auto-from-filenames

# Hoặc test trước
python test_text_normalizer.py
python test_structure_chunking.py
```

## 🧪 Test Scripts

### test_text_normalizer.py
```bash
python test_text_normalizer.py
```

**Options:**
- [1] Test với Sample Text (nhanh)
- [2] Test với PDF Files (data/knowledge_base)
- [3] Tất cả tests

**Output:**
- Hiển thị text trước/sau normalize
- So sánh số chunks với/không normalize
- Phân tích headers được phát hiện

### test_structure_chunking.py
```bash
python test_structure_chunking.py
```

**Options:**
- [1] Test với Sample Markdown
- [2] Test với Real PDF Files
- [3] Cả hai

**Output:**
- Preview chunks
- Context injection analysis
- Headers count

## 📊 Ví Dụ Thực Tế

### Input (Text thô):
```
THONG TIN TRUNG TAM

1. Cac khoa hoc

1.1. Tieng Anh co ban
Noi dung...

1.2. Tieng Anh nang cao
Noi dung...

2. Hoc phi

2.1. Chinh sach giam gia
Noi dung...
```

### Output (Sau normalize + chunking):

**Chunk 1:**
```
## THONG TIN TRUNG TAM
```

**Chunk 2:**
```
## 1. Cac khoa hoc
### 1.1. Tieng Anh co ban
Noi dung...
```

**Chunk 3:**
```
## 1. Cac khoa hoc
### 1.2. Tieng Anh nang cao
Noi dung...
```

**Chunk 4:**
```
## 2. Hoc phi
### 2.1. Chinh sach giam gia
Noi dung...
```

## 💡 Ưu Điểm

### So với Fixed-Size Chunking:
| Tiêu chí | Fixed-Size | Structure-Based |
|----------|------------|-----------------|
| **Tính nhất quán** | ⚠️ Cắt tùy tiện | ✅ Theo cấu trúc |
| **Ngữ cảnh** | ⚠️ Có thể mất | ✅ Bảo toàn hoàn toàn |
| **Chất lượng RAG** | Trung bình | Cao hơn nhiều |
| **Không trộn lẫn** | ❌ | ✅ Mỗi section riêng biệt |

### Khi nào dùng Structure-Based:
- ✅ Tài liệu có cấu trúc rõ ràng (PDF, Word, Markdown)
- ✅ Cần giữ nguyên ngữ cảnh từng phần
- ✅ Câu hỏi phức tạp cần context dài
- ✅ Muốn tránh trộn lẫn nội dung khác nhau

## 🔧 Code Structure

```
src/
├── chunking_strategies.py
│   ├── TextNormalizer          # Auto-detect và thêm ##/###
│   ├── DocumentBasedParser     # Structure-based chunking
│   └── get_node_parser()       # Factory function
└── config.py                   # Cấu hình

test_text_normalizer.py         # Test normalizer
test_structure_chunking.py      # Test chunking
```

## 📝 TextNormalizer Details

### Patterns Detected:

1. **Level 1 Number Pattern:**
   - Regex: `^(\d+)\.\s+(.+)$`
   - Example: `1. Giới thiệu`
   - Điều kiện: Title ≥ 5 chars, bắt đầu chữ hoa

2. **Level 1 Uppercase Pattern:**
   - Regex: `^([A-ZÀ-Ỹ][A-ZÀ-Ỹ\s]{2,})$`
   - Example: `THÔNG TIN CHUNG`
   - Điều kiện: ≥ 2 từ, 10-100 chars, không có số

3. **Level 2 Pattern:**
   - Regex: `^(\d+)\.(\d+)\.\s+(.+)$`
   - Example: `1.1. Học phí cơ bản`
   - Điều kiện: Title ≥ 3 chars, bắt đầu chữ hoa

### Edge Cases Handled:

```python
# ❌ KHÔNG normalize (list items)
"1. là học phí"  # Quá ngắn
"2. dong tien"   # Không hoa

# ✅ NORMALIZE (headers thật)
"1. Giới thiệu khóa học"  # Dài, chữ hoa
"THÔNG TIN CHUNG"         # In hoa toàn bộ
```

## 🎯 Best Practices

1. **Luôn test trước:**
   ```bash
   python test_text_normalizer.py  # Option 2
   ```

2. **Kiểm tra output:**
   - Xem có headers bị miss không
   - Xem có false positives không

3. **Tune parameters nếu cần:**
   ```python
   DOC_BASED_MIN_CHUNK_SIZE = 200  # Tăng nếu chunks quá nhỏ
   DOC_BASED_MAX_CHUNK_SIZE = 1500 # Giảm nếu chunks quá lớn
   ```

4. **Chạy ingest sau khi hài lòng:**
   ```bash
   python ingest.py --auto-from-filenames
   ```

## 🐛 Troubleshooting

**Q: Headers không được phát hiện?**
- Check pattern có match không (độ dài, chữ hoa)
- Thử adjust regex trong TextNormalizer

**Q: Quá nhiều false positives?**
- Tăng độ dài tối thiểu trong conditions
- Thêm điều kiện kiểm tra stricter

**Q: Chunks vẫn bị trộn lẫn?**
- Check xem text có được normalize chưa
- Verify auto_normalize=True

---

**Lưu ý:** Sau khi thay đổi chunking strategy, nhớ chạy lại ingest để áp dụng cho toàn bộ dữ liệu!

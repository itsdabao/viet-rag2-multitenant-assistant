"""
Chunking Strategies cho việc chia nhỏ tài liệu.
Hỗ trợ nhiều chiến lược: Fixed-Size, Document-Based, Semantic.
"""

import re
from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import (
    SentenceSplitter,
    NodeParser,
)
from llama_index.core.schema import BaseNode
from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DOC_BASED_MIN_CHUNK_SIZE,
    DOC_BASED_MAX_CHUNK_SIZE,
)


class TextNormalizer:
    """
    Normalize văn bản thô thành Markdown format với headers ##/###.
    
    Tự động phát hiện:
    - Level 1 Header (##): Dòng bắt đầu với số + dấu chấm (1., 2.) HOẶC chữ in hoa toàn bộ
    - Level 2 Header (###): Dòng bắt đầu với số cấp 2 (1.1., 5.2.)
    - Fix malformed headers: ##1.Title -> ## 1. Title
    
    Cẩn thận không nhầm với danh sách trong đoạn văn.
    """
    
    def __init__(self):
        # Pattern cho Level 1: Số đơn + dấu chấm ở đầu dòng
        # Ví dụ: "1. Giới thiệu", "2. Học phí"
        self.level1_number_pattern = re.compile(r'^(\d+)\.\s+(.+)$', re.MULTILINE)
        
        # Pattern cho Level 1: Dòng viết HOA toàn bộ
        # Ví dụ: "THÔNG TIN CHUNG", "HỌC PHÍ VÀ ƯU ĐÃI"
        # Yêu cầu: Ít nhất 2 từ, toàn bộ chữ in hoa, không có số
        self.level1_uppercase_pattern = re.compile(
            r'^([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ\s]{2,})$',
            re.MULTILINE
        )
        
        # Pattern cho Level 2: Số cấp 2 + dấu chấm
        # Ví dụ: "1.1. Học phí cơ bản", "5.2. Giáo viên nước ngoài"
        self.level2_pattern = re.compile(r'^(\d+)\.(\d+)\.\s+(.+)$', re.MULTILINE)
        
        # Patterns để fix malformed headers (không có space)
        # Ví dụ: "##1.Title" -> "## 1. Title", "#TITLE" -> "# TITLE"
        self.malformed_level1_pattern = re.compile(r'^##(\d+)\.(.+)$', re.MULTILINE)  # ##1.Title
        self.malformed_level2_pattern = re.compile(r'^###(\d+)\.(\d+)\.(.+)$', re.MULTILINE)  # ###1.1.Title
        self.malformed_hash_pattern = re.compile(r'^#([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ].+)$', re.MULTILINE)  # #TITLE
    
    def normalize(self, text: str) -> str:
        """
        Normalize text bằng cách thêm Markdown headers.
        
        Args:
            text: Văn bản thô
            
        Returns:
            str: Văn bản đã được normalize với ##/###
        """
        # Bước 0: Fix malformed headers trước (##1.Title -> ## 1. Title)
        text = self._fix_malformed_headers(text)
        
        # Bước 1: Thêm ### cho Level 2 headers (làm trước để không bị nhầm với Level 1)
        text = self._add_level2_headers(text)
        
        # Bước 2: Thêm ## cho Level 1 headers (số)
        text = self._add_level1_number_headers(text)
        
        # Bước 3: Thêm ## cho Level 1 headers (chữ in hoa)
        text = self._add_level1_uppercase_headers(text)
        
        return text
    
    def _fix_malformed_headers(self, text: str) -> str:
        """Fix các headers không chuẩn như ##1.Title, ###1.1.Title"""
        
        # Fix ##1.Title -> ## 1. Title
        def fix_level1(match):
            num = match.group(1)
            title = match.group(2).strip()
            return f"## {num}. {title}"
        
        text = self.malformed_level1_pattern.sub(fix_level1, text)
        
        # Fix ###1.1.Title -> ### 1.1. Title  
        def fix_level2(match):
            major = match.group(1)
            minor = match.group(2)
            title = match.group(3).strip()
            return f"### {major}.{minor}. {title}"
        
        text = self.malformed_level2_pattern.sub(fix_level2, text)
        
        # Fix #TITLE -> # TITLE (nếu không có space sau #)
        def fix_hash(match):
            title = match.group(1)
            return f"# {title}"
        
        text = self.malformed_hash_pattern.sub(fix_hash, text)
        
        return text
    
    def _add_level2_headers(self, text: str) -> str:
        """Thêm ### cho các dòng như "1.1. Title", "5.2. Title"."""
        def replace_fn(match):
            major = match.group(1)  # Số chính (1, 5)
            minor = match.group(2)  # Số phụ (1, 2)
            title = match.group(3)  # Tiêu đề
            
            # Kiểm tra xem có phải header thật không
            # Header thật thường có title dài hơn và bắt đầu bằng chữ hoa
            if len(title) >= 3 and title[0].isupper():
                return f"### {major}.{minor}. {title}"
            else:
                # Giữ nguyên nếu không phải header
                return match.group(0)
        
        return self.level2_pattern.sub(replace_fn, text)
    
    def _add_level1_number_headers(self, text: str) -> str:
        """Thêm ## cho các dòng như "1. Title", "2. Title"."""
        def replace_fn(match):
            number = match.group(1)  # Số thứ tự
            title = match.group(2)   # Tiêu đề
            
            # Kiểm tra điều kiện:
            # 1. Title phải đủ dài (ít nhất 5 ký tự)
            # 2. Bắt đầu bằng chữ hoa
            # 3. Không có ### ở đầu (tránh xử lý lại Level 2)
            if (len(title) >= 5 and 
                title[0].isupper() and 
                not title.startswith('###')):
                return f"## {number}. {title}"
            else:
                # Giữ nguyên nếu không phải header (có thể là list item)
                return match.group(0)
        
        return self.level1_number_pattern.sub(replace_fn, text)
    
    def _add_level1_uppercase_headers(self, text: str) -> str:
        """Thêm ## cho các dòng viết HOA toàn bộ như "THÔNG TIN CHUNG"."""
        def replace_fn(match):
            line = match.group(1).strip()
            
            # Kiểm tra điều kiện:
            # 1. Không chứa số
            # 2. Có ít nhất 2 từ (ngăn cách bởi space)
            # 3. Độ dài hợp lý (10-100 chars)
            # 4. Chưa có ## ở đầu
            if (not any(char.isdigit() for char in line) and
                len(line.split()) >= 2 and
                10 <= len(line) <= 100 and
                not line.startswith('##')):
                return f"## {line}"
            else:
                return match.group(0)
        
        return self.level1_uppercase_pattern.sub(replace_fn, text)
    
    def detect_has_structure(self, text: str) -> bool:
        """
        Kiểm tra xem text có cấu trúc header không.
        
        Returns:
            bool: True nếu text có ít nhất 1 pattern header
        """
        has_level1_number = bool(self.level1_number_pattern.search(text))
        has_level1_uppercase = bool(self.level1_uppercase_pattern.search(text))
        has_level2 = bool(self.level2_pattern.search(text))
        has_markdown = '##' in text or '###' in text
        
        return has_level1_number or has_level1_uppercase or has_level2 or has_markdown



class DocumentBasedParser(NodeParser):
    """
    Structure-Based Parser: Chia tài liệu dựa trên cấu trúc Markdown (## và ###).
    
    Nguyên tắc:
    - Gặp ## (Section) → Bắt buộc cắt chunk mới
    - Gặp ### (Sub-section) → Tách riêng nếu đủ dài
    - Context Injection: Sub-section ### sẽ có tiêu đề cha ## ở đầu
    """

    def __init__(
        self,
        min_chunk_size: int = DOC_BASED_MIN_CHUNK_SIZE,
        max_chunk_size: int = DOC_BASED_MAX_CHUNK_SIZE,
        auto_normalize: bool = True,  # Tự động normalize text
        **kwargs
    ):
        super().__init__(**kwargs)
        # Bypass Pydantic validation
        object.__setattr__(self, '_min_chunk_size', min_chunk_size)
        object.__setattr__(self, '_max_chunk_size', max_chunk_size)
        object.__setattr__(self, '_auto_normalize', auto_normalize)
        object.__setattr__(self, '_normalizer', TextNormalizer() if auto_normalize else None)
    
    @property
    def min_chunk_size(self) -> int:
        return getattr(self, '_min_chunk_size', DOC_BASED_MIN_CHUNK_SIZE)
    
    @property
    def max_chunk_size(self) -> int:
        return getattr(self, '_max_chunk_size', DOC_BASED_MAX_CHUNK_SIZE)
    
    @property
    def auto_normalize(self) -> bool:
        return getattr(self, '_auto_normalize', True)
    
    @property
    def normalizer(self):
        return getattr(self, '_normalizer', None)

    def _parse_nodes(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs
    ) -> List[BaseNode]:
        """Parse documents into nodes based on Markdown structure."""
        all_nodes = []

        for node in nodes:
            # Lấy text từ node
            text = node.get_content(metadata_mode="none")
            
            # Auto-normalize text nếu được bật
            # LUÔN chạy normalize để:
            # 1. Fix malformed headers (##1.Title -> ## 1. Title)
            # 2. Thêm headers mới nếu chưa có
            if self.auto_normalize and self.normalizer:
                normalized_text = self.normalizer.normalize(text)
                text = normalized_text
                if show_progress:
                    print(f"[TextNormalizer] Đã normalize text ({len(text)} chars)")
            
            # Phân tích cấu trúc và tạo chunks
            chunks = self._structure_based_chunking(text)
            
            # Tạo nodes mới từ chunks
            for chunk_text in chunks:
                if not chunk_text.strip():
                    continue
                    
                from llama_index.core.schema import TextNode
                new_node = TextNode(
                    text=chunk_text,
                    metadata=node.metadata.copy() if hasattr(node, 'metadata') else {},
                )
                all_nodes.append(new_node)

        return all_nodes

    def _structure_based_chunking(self, text: str) -> List[str]:
        """
        Chia text dựa trên cấu trúc Markdown ## và ###.
        
        Returns:
            List[str]: Danh sách chunks
        """
        import re
        
        chunks = []
        
        # Tìm tất cả các sections (##) và sub-sections (###)
        # Pattern: bắt đầu dòng, có ## hoặc ###, theo sau là text
        section_pattern = r'^(#{2,3})\s+(.+?)$'
        
        # Split text thành các dòng
        lines = text.split('\n')
        
        current_section = None  # Lưu tiêu đề ## hiện tại
        current_subsection = None  # Lưu tiêu đề ### hiện tại
        current_content = []  # Nội dung của section/subsection hiện tại
        
        i = 0
        while i < len(lines):
            line = lines[i]
            match = re.match(section_pattern, line)
            
            if match:
                header_level = len(match.group(1))  # 2 hoặc 3
                header_text = match.group(2).strip()
                
                if header_level == 2:  # Gặp Section mới (##)
                    # Lưu subsection cũ nếu có
                    if current_subsection and current_content:
                        chunk = self._create_chunk_with_context(
                            current_section, current_subsection, current_content
                        )
                        if chunk:
                            chunks.append(chunk)
                    # Lưu section cũ nếu có (không có subsection con)
                    elif current_section and current_content:
                        chunk = self._create_chunk_simple(current_section, current_content)
                        if chunk:
                            chunks.append(chunk)
                    
                    # Bắt đầu section mới
                    current_section = f"## {header_text}"
                    current_subsection = None
                    current_content = []
                
                elif header_level == 3:  # Gặp Sub-section mới (###)
                    # Lưu subsection cũ nếu có
                    if current_subsection and current_content:
                        chunk = self._create_chunk_with_context(
                            current_section, current_subsection, current_content
                        )
                        if chunk:
                            chunks.append(chunk)
                    
                    # Bắt đầu subsection mới
                    current_subsection = f"### {header_text}"
                    current_content = []
            
            else:
                # Nội dung thông thường
                if line.strip():  # Bỏ qua dòng trống
                    current_content.append(line)
            
            i += 1
        
        # Lưu chunk cuối cùng
        if current_subsection and current_content:
            chunk = self._create_chunk_with_context(
                current_section, current_subsection, current_content
            )
            if chunk:
                chunks.append(chunk)
        elif current_section and current_content:
            chunk = self._create_chunk_simple(current_section, current_content)
            if chunk:
                chunks.append(chunk)
        
        # Fallback: Nếu không có header nào, chia theo paragraph thông thường
        if not chunks:
            chunks = self._fallback_paragraph_chunking(text)
        
        return chunks

    def _create_chunk_with_context(
        self, 
        section_header: str, 
        subsection_header: str, 
        content_lines: List[str]
    ) -> str:
        """
        Tạo chunk cho subsection với context injection (chèn tiêu đề cha).
        
        Args:
            section_header: Tiêu đề ## cha
            subsection_header: Tiêu đề ### con
            content_lines: Nội dung của subsection
            
        Returns:
            str: Chunk với context
        """
        if not content_lines:
            return ""
        
        # Context injection: Thêm tiêu đề cha vào đầu
        parts = []
        if section_header:
            parts.append(section_header)
        if subsection_header:
            parts.append(subsection_header)
        
        # Nội dung
        content = '\n'.join(content_lines).strip()
        parts.append(content)
        
        chunk = '\n'.join(parts)
        
        # Kiểm tra kích thước
        if len(chunk) > self.max_chunk_size:
            # Nếu quá dài, cần chia nhỏ
            return self._split_oversized_chunk(chunk, section_header, subsection_header)
        
        return chunk

    def _create_chunk_simple(self, header: str, content_lines: List[str]) -> str:
        """
        Tạo chunk đơn giản cho section không có subsection.
        
        Args:
            header: Tiêu đề ## hoặc ###
            content_lines: Nội dung
            
        Returns:
            str: Chunk
        """
        if not content_lines:
            return ""
        
        content = '\n'.join(content_lines).strip()
        chunk = f"{header}\n{content}"
        
        # Kiểm tra kích thước
        if len(chunk) > self.max_chunk_size:
            return self._split_oversized_chunk(chunk, header, None)
        
        return chunk

    def _split_oversized_chunk(
        self, 
        chunk: str, 
        section_header: str = None, 
        subsection_header: str = None
    ) -> str:
        """
        Chia chunk quá lớn thành nhiều phần nhỏ hơn.
        Chỉ trả về phần đầu tiên, phần còn lại sẽ bị mất (trade-off).
        
        TODO: Có thể cải tiến để trả về list chunks thay vì 1 chunk.
        """
        import re
        
        # Thử chia theo câu
        sentences = re.split(r'(?<=[.!?])\s+', chunk)
        
        # Xây dựng chunk mới trong giới hạn
        parts = []
        current_len = 0
        
        # Thêm headers trước
        if section_header:
            parts.append(section_header)
            current_len += len(section_header) + 1
        if subsection_header:
            parts.append(subsection_header)
            current_len += len(subsection_header) + 1
        
        # Thêm câu cho đến khi đầy
        for sent in sentences:
            if current_len + len(sent) + 1 <= self.max_chunk_size:
                parts.append(sent)
                current_len += len(sent) + 1
            else:
                break
        
        return '\n'.join(parts) if parts else chunk[:self.max_chunk_size]

    def _fallback_paragraph_chunking(self, text: str) -> List[str]:
        """
        Fallback: Nếu không có cấu trúc Markdown, chia theo paragraph thông thường.
        """
        import re
        
        # Chia theo paragraph (dấu xuống dòng đôi)
        paragraphs = re.split(r'\n\s*\n+', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if not current_chunk:
                current_chunk = para
            elif len(current_chunk) + len(para) + 2 <= self.max_chunk_size:
                current_chunk += "\n\n" + para
            else:
                chunks.append(current_chunk)
                current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks


def get_node_parser(strategy: str = "fixed_size") -> NodeParser:
    """
    Factory function để lấy node parser theo strategy.
    
    Args:
        strategy: "fixed_size", "document_based", hoặc "semantic"
    
    Returns:
        NodeParser instance phù hợp
    """
    if strategy == "fixed_size":
        return SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
    
    elif strategy == "document_based":
        return DocumentBasedParser(
            min_chunk_size=DOC_BASED_MIN_CHUNK_SIZE,
            max_chunk_size=DOC_BASED_MAX_CHUNK_SIZE
        )
    
    elif strategy == "semantic":
        # Semantic splitting requires embedding model, có thể implement sau
        # Fallback to fixed_size for now
        print("⚠️ Semantic chunking chưa được implement, dùng fixed_size thay thế.")
        return SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
    
    else:
        raise ValueError(
            f"Unknown chunking strategy: {strategy}. "
            f"Supported: 'fixed_size', 'document_based', 'semantic'"
        )

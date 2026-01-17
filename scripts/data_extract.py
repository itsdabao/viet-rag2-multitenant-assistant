import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    EasyOcrOptions,  # Đổi từ Tesseract sang EasyOCR
    TableFormerMode
)
from docling.document_converter import DocumentConverter, PdfFormatOption

input_path = r"D:\AI_Agent\data\knowledge_base\no_prprocessed_pdf\EvasProfile.pdf"
output_dir = r"D:\AI_Agent\data\knowledge_base\preprocessed_markdown"
os.makedirs(output_dir, exist_ok=True)

def process_evas_to_markdown(file_path):
    pipeline_options = PdfPipelineOptions()
    
    # Kích hoạt EasyOCR - Nhận diện tiếng Việt rất tốt
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = EasyOcrOptions()
    
    # Giữ cấu hình Table SOTA để lấy học phí chính xác
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.table_structure_options.do_cell_matching = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    print(f"--- Đang bắt đầu xử lý SOTA (EasyOCR) cho: {os.path.basename(file_path)} ---")
    result = converter.convert(file_path)
    
    md_content = result.document.export_to_markdown()
    save_path = os.path.join(output_dir, Path(file_path).stem + ".md")
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"--- Thành công! File Markdown lưu tại: {save_path} ---")

if __name__ == "__main__":
    process_evas_to_markdown(input_path)
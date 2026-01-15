from typing import List, Dict

def build_prompt(query: str, examples: List[Dict[str, str]], retrieved_texts: List[str], history: List[Dict[str, str]] | None = None) -> str:
    # 1. Load Template (Giả sử file chứa các placeholder: {chat_history}, {context_str}, {user_query})
    system_prompt_template = _load_system_prompt(SYSTEM_PROMPT_PATH) 
    
    # 2. Xử lý Context (Gộp list thành 1 chuỗi string để nhét vào XML)
    if retrieved_texts:
        # Format từng doc và join lại
        context_str = "\n".join([f"[Document {i}]: {text}" for i, text in enumerate(retrieved_texts, 1)])
    else:
        context_str = "Không tìm thấy tài liệu liên quan."

    # 3. Xử lý History (Nếu có)
    history_str = ""
    if history:
        # Render history thành string, ví dụ: "User: A\nAI: B"
        # Hàm _render_history của bạn cần trả về string thay vì list, hoặc join ở đây
        hist_lines = _render_history(history) 
        if isinstance(hist_lines, list):
            history_str = "\n".join(hist_lines)
        else:
            history_str = str(hist_lines)
    else:
        history_str = "Chưa có lịch sử hội thoại."

    # 4. Xử lý Examples (Few-shot)
    # Examples được đưa cứng vào trong file system prompt luôn.

    # 5. Injection (Điền vào chỗ trống)
    # Dùng .format() hoặc f-string an toàn để thay thế placeholder
    final_prompt = system_prompt_template.format(
        chat_history=history_str,
        context_str=context_str,
        user_query=query
    )
    
    return final_prompt
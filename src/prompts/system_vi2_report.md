# 📄 Phân Tích Tối Ưu Hóa System Prompt: Trợ Lý Ảo Trung Tâm Tiếng Anh (RAG Pipeline)

Tài liệu này trình bày chi tiết cách khắc phục các nhược điểm của Prompt cũ, các kỹ thuật Prompt Engineering đã áp dụng và lý do tại sao phiên bản mới (V2) mang lại hiệu quả vượt trội cho mô hình **Gemini 2.0 Flash**.

---

## 1. Bảng So Sánh & Kỹ Thuật Áp Dụng

| Vấn đề (Prompt Cũ) | Giải pháp (Prompt Mới) | Kỹ thuật Prompt Engineering | Ưu điểm (Lý do) |
| :--- | :--- | :--- | :--- |
| **Cấu trúc lỏng lẻo:** Dùng Markdown (`##`) lẫn lộn giữa chỉ dẫn và dữ liệu đầu vào. | **Cấu trúc rõ ràng:** Sử dụng thẻ XML (`<tag>`) để bao bọc từng phần riêng biệt. | **XML Tagging / Delimiters** (Phân tách ngữ nghĩa) | Giúp model phân biệt rõ đâu là "Lệnh hệ thống", đâu là "Dữ liệu tra cứu" (Context), giảm thiểu việc model bị nhầm lẫn nội dung. |
| **Dễ bị Hallucination (Bịa đặt):** Chỉ nói "Dữ liệu là chân lý" một cách chung chung. | **Grounding chặt chẽ:** Ép buộc model chỉ được nhìn vào `<retrieved_context>` và quy định hành vi cụ thể khi không thấy thông tin. | **Context Grounding & Negative Constraints** (Ràng buộc phủ định) | Triệt tiêu ảo giác. Model biết chính xác giới hạn kiến thức của nó nằm ở đâu trong đoạn text được cung cấp. |
| **Hội thoại như "Thẩm vấn":** Model có xu hướng hỏi dồn dập nhiều thông tin cùng lúc để hoàn thành task. | **Chiến thuật "Give & Take":** Quy định "Không hỏi quá 1 câu/lượt" và phải cung cấp giá trị trước khi đòi thông tin. | **Constraint-Based Prompting** (Ràng buộc hành vi) | Tạo trải nghiệm người dùng tự nhiên, thân thiện hơn, tránh làm khách hàng cảm thấy bị làm phiền. |
| **Trigger cứng nhắc:** Chỉ xuất dữ liệu khi có *đủ toàn bộ* thông tin (Tên, Tuổi, SĐT...). | **Trigger linh hoạt:** Ưu tiên số điện thoại (Primary Key), các trường khác cho phép `Unknown`. | **Logical Relaxation** (Nới lỏng logic) | Tăng tỷ lệ chuyển đổi (Conversion Rate). Tránh mất Lead chỉ vì khách hàng lười cung cấp thông tin phụ. |
| **Xử lý thiếu thông tin thụ động:** Chỉ biết "Xin lỗi". | **Xử lý chủ động:** Biến lời xin lỗi thành cơ hội lấy số điện thoại (Call-to-Action). | **Instruction Tuning / Role-playing** | Chuyển đổi tình huống tiêu cực (thiếu data) thành tích cực (cơ hội sales), đúng với mục tiêu kinh doanh. |

---

## 2. Chi Tiết Các Cải Tiến Quan Trọng

### 2.1. Sử dụng XML Tags thay vì Markdown
Các mô hình LLM hiện đại (đặc biệt là Gemini và Claude) được huấn luyện để hiểu cấu trúc XML rất tốt.
* **Cũ:**
    ```text
    ## Dữ liệu
    [Nội dung RAG]
    ```
* **Mới:**
    ```xml
    <retrieved_context>
    [Nội dung RAG]
    </retrieved_context>
    ```
    > **Lý do:** Ngăn chặn việc model nhầm lẫn nội dung trong tài liệu tiếng Anh (ví dụ tài liệu có chứa các dòng hướng dẫn) với lệnh của hệ thống.

### 2.2. Kỹ thuật Few-Shot (Mô phỏng ví dụ)
Thay vì chỉ mô tả trừu tượng "hãy khéo léo", Prompt V2 đưa ra ví dụ cụ thể về hành vi mong muốn.
* **Cơ chế:** Cung cấp mẫu "Ví dụ Tốt" và "Ví dụ Xấu".
* **Tác dụng:** Giúp model căn chỉnh Tone & Mood (giọng điệu) chính xác ngay từ đầu mà không cần suy diễn.

### 2.3. Định dạng đầu ra cho Hệ thống (System Output)
Tối ưu hóa block `LEAD_DATA` để dễ dàng cho Code Python xử lý hậu kỳ (Post-processing).
* **Quy tắc:** Sử dụng Regex pattern dễ bắt.
* **Cải tiến:** `Họ tên: [Value/Unknown]` -> Cho phép giá trị `Unknown` giúp code không bị lỗi khi thiếu trường dữ liệu, đảm bảo pipeline luôn chạy mượt mà.

---

## 3. Kết Luận

Việc chuyển đổi từ Prompt dạng văn bản thông thường sang **Structured Prompt (Prompt có cấu trúc)** yêu cầu thay đổi code hàm build_prompt trong file incontext_ralm.py


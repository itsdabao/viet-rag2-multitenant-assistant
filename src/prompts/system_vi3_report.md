# Báo Cáo Tối Ưu Hóa System Prompt (RAG Pipeline)

Tài liệu này trình bày chi tiết cách khắc phục các nhược điểm của Prompt cũ, các kỹ thuật Prompt Engineering đã áp dụng và lý do tại sao phiên bản mới mang lại hiệu quả vượt trội.
---

## 1. Bảng So Sánh & Khắc Phục Nhược Điểm

| Tiêu chí | Instruction Cũ (Nhược điểm) | Instruction Mới (Giải pháp) | Tại sao Mới tốt hơn? |
| :--- | :--- | :--- | :--- |
| **Liên kết Dữ liệu (Binding)** | Chỉ yêu cầu chung chung "Dựa trên tài liệu". Không chỉ rõ cấu trúc dữ liệu đầu vào. | Gọi tên chính xác các phần dữ liệu mà code Python tạo ra: **History**, **Examples**, **Retrieved Knowledge ([Doc i])**. | Giúp Model phân biệt rõ đâu là *Kiến thức nền (Pre-trained)* và đâu là *Dữ liệu RAG*, giảm thiểu Hallucination (Ảo giác). |
| **Xử lý Lịch sử (History)** | Không đề cập đến việc kiểm tra lịch sử hội thoại. | Yêu cầu rõ ràng: "Luôn xem xét History... tránh hỏi lại thông tin đã cung cấp". | Ngăn chặn việc Chatbot hỏi đi hỏi lại tên/tuổi của khách hàng dù họ đã nói ở các câu trước. |
| **Thu thập Lead (Sales)** | Hướng dẫn thụ động: "Hỏi khéo léo từng thông tin". Model dễ bị quên hoặc hỏi dồn dập. | Áp dụng quy trình 3 bước: **Phân tích -> Kiểm tra thiếu -> Hành động**. Chỉ hỏi 1 câu/lần. | Biến việc hỏi thông tin thành một quy trình logic (Logic flow), giúp cuộc hội thoại tự nhiên và ít gây khó chịu hơn. |
| **Cấu trúc Chỉ dẫn** | Các quy tắc viết lẫn lộn, thiếu điểm nhấn mạnh. | Sử dụng Phân tách rõ ràng (Delimiters): `##`, `QUY TẮC BẮT BUỘC`, `KHI VÀ CHỈ KHI`. | Model (Gemini 2.0) ưu tiên tuân thủ các chỉ dẫn có cấu trúc rõ ràng, đặc biệt trong các ngữ cảnh dài (Long context). |
| **Tận dụng Ví dụ (Few-shot)** | Không nhắc đến các ví dụ mẫu (Examples). | Có mục nhắc nhở tham khảo `Examples` để học văn phong trả lời. | Tận dụng tối đa dữ liệu `examples` được truyền vào từ hàm `build_prompt`, giúp câu trả lời chuẩn xác hơn về giọng điệu. |

---

## 2. Các Kỹ Thuật Prompt Engineering Đã Áp Dụng

### 1. Context Grounding (Neo Ngữ Cảnh)
* **Cách áp dụng:** Định nghĩa rõ ràng khu vực dữ liệu `Retrieved Knowledge` và gắn thẻ `[Doc i]`.
* **Mục đích:** Khóa chặt phạm vi trả lời của AI vào dữ liệu truy xuất được từ Vector Database (Qdrant), ngăn AI sáng tác ra các chính sách khuyến mãi không có thật.

### 2. Chain-of-Thought (Chuỗi Suy Luận)
* **Cách áp dụng:** Thay vì ra lệnh "Hãy lấy số điện thoại", prompt mới hướng dẫn AI *cách suy nghĩ*: "Bước 1: Kiểm tra History -> Bước 2: Xem còn thiếu trường nào -> Bước 3: Mới đặt câu hỏi".
* **Mục đích:** Tăng độ thông minh cho Agent. AI sẽ tự hiểu rằng nếu khách đã nói "Mình tên Lan, sinh năm 95" thì chỉ cần hỏi Số điện thoại và Nhu cầu, không hỏi lại tên/tuổi.

### 3. Instruction Delimiting (Phân Tách Chỉ Dẫn)
* **Cách áp dụng:** Sử dụng triệt để Markdown Headers (`##`), In đậm (`**`), và Block Code cho output (`LEAD_DATA`).
* **Mục đích:** Giúp LLM phân tích cú pháp (parse) prompt tốt hơn. Nó hiểu rõ đâu là vai trò (Persona), đâu là luật cấm (Negative constraint), và đâu là định dạng đầu ra mong muốn.

### 4. Meta-Instruction cho Few-Shot Learning
* **Cách áp dụng:** Thêm hướng dẫn để model chú ý đến phần `Examples` được code Python append vào.
* **Mục đích:** Instruction cũ bỏ phí phần `examples` trong code. Instruction mới biến nó thành công cụ để tinh chỉnh giọng văn (Tone of Voice) cho giống nhân viên thật.

---

## 3. Bổ sung Kỹ Thuật "Static Few-Shot Prompting" trong Instruction

Một thay đổi quan trọng trong phiên bản mới là việc bổ sung phần `## Ví dụ Hành vi` trực tiếp vào Instruction, thay vì chỉ phụ thuộc vào danh sách `examples` động từ code Python.

### Chi tiết thay đổi và Đóng góp

| Hạng mục | Instruction Cũ | Instruction Mới (Có Static Few-shot) | Đóng góp vào hiệu suất (Impact) |
| :--- | :--- | :--- | :--- |
| **Xử lý ngữ cảnh ẩn** | Model thường lúng túng khi khách hỏi câu mơ hồ (VD: "Giá sao?"). Thường trả lời liệt kê dài dòng. | Cung cấp mẫu đối thoại (Tình huống 1) dạy model cách **Hỏi ngược (Clarifying Question)** để lọc nhu cầu trước khi báo giá. | Giúp cuộc hội thoại mang tính tư vấn hai chiều, tránh việc "trả bài" thông tin một cách máy móc. |
| **Khai thác History** | Model hay quên tên khách hoặc hỏi lại thông tin đã có. | Cung cấp mẫu (Tình huống 2) minh họa cụ thể việc trích xuất tên/tuổi từ `History` để chào hỏi. | Tăng trải nghiệm cá nhân hóa (Personalization). Khách hàng cảm thấy được quan tâm vì AI nhớ thông tin cũ. |
| **Định dạng Output** | Chỉ mô tả bằng lời văn về `LEAD_DATA`. Model dễ in sai hoặc thêm lời dẫn thừa. | Cung cấp ví dụ trực quan (Visual Example) về block code `LEAD_DATA` trong ngữ cảnh thực tế. | **Consistency:** Đảm bảo định dạng đầu ra tuyệt đối chính xác để hệ thống backend (Regex) bắt được dữ liệu Lead. |

### Tại sao cần cả "Static Examples" (trong Instruction) và "Dynamic Examples" (trong Code)?

* **Static Examples (Trong Instruction):** Dùng để dạy **Logic hội thoại** (Cách chào, cách hỏi lead, cách xử lý từ chối). Những logic này là bất biến, cần "ghim" cứng vào não model.
* **Dynamic Examples (Trong Code/RAG):** Dùng để cung cấp **Kiến thức nghiệp vụ** (Giá tiền, lịch học cụ thể). Những thông tin này thay đổi tùy theo câu hỏi của khách nên cần retrieve động.

-> **Kết luận:** Sự kết hợp này tạo ra một "Hybrid Prompt" giúp Gemini 2.0 Flash vừa thông minh trong cách ứng xử, vừa chính xác trong nội dung tư vấn. **Có đi kèm file test_case.json gồm 20 few-shot prompting để mô hình gọi tới cho Dynamic Examples**.

---

## 4. Kết Luận: Tại sao bản mới phù hợp?

1.  **Tương thích Code:** Prompt mới được thiết kế "đo ni đóng giày" cho hàm `build_prompt`. Nó biết chính xác code sẽ nối chuỗi theo thứ tự nào (System -> History -> Examples -> Retrieval) và hướng dẫn model xử lý từng phần đó.

2.  **Giảm Tỷ lệ Lỗi (Robustness):** Bằng cách yêu cầu model kiểm tra `History` trước khi hỏi, hệ thống sẽ tránh được các tình huống "ngớ ngẩn" (như hỏi tên khách hàng 2 lần), tăng trải nghiệm người dùng (UX).

3.  **Áp dụng các kỹ thuật Prompt Engineering:** Các kỹ thuật Context Grounding, Chain-of-Thought, Instruction Delimiting và Hybrid Few-shot Prompting được kết hợp một cách có chủ đích, không rời rạc. Điều này giúp model không chỉ “trả lời đúng”, mà còn **suy nghĩ đúng quy trình**, biết khi nào cần hỏi, khi nào cần tư vấn và khi nào cần xuất dữ liệu cho hệ thống.

4.  **Dữ liệu đầu ra sạch (Structured Output):** Định dạng `LEAD_DATA` được quy định chặt chẽ hơn, đảm bảo hệ thống backend có thể Regex để lấy thông tin khách hàng chính xác 100%.
import pandas as pd
import os

data = [
    # General Info
    ["Trung tâm ở đâu?", "Dạ trung tâm có 2 cơ sở: 123 Nguyễn Văn Linh, Đà Nẵng và 456 Lê Duẩn, Hà Nội ạ."],
    ["Giờ làm việc của trung tâm thế nào?", "Dạ bên em mở cửa từ 8h00 - 21h00 tất cả các ngày trong tuần ạ."],
    ["Có chỗ để xe không?", "Dạ có ạ, trung tâm có hầm để xe rộng rãi và miễn phí cho học viên ạ."],
    
    # Courses & Tuition
    ["Học phí khóa TOEIC bao nhiêu?", "Dạ khóa TOEIC căn bản học phí là 3.500.000đ/khóa (3 tháng). Đang có ưu đãi giảm 500k cho đăng ký sớm ạ."],
    ["Khóa IELTS học phí thế nào?", "Dạ IELTS Foundation học phí 5.000.000đ/khóa. IELTS Intensive là 9.000.000đ/khóa ạ."],
    ["Khóa Giao tiếp bao nhiêu tiền?", "Dạ khóa Giao tiếp phản xạ học phí 4.200.000đ/khóa (2.5 tháng) ạ."],
    ["Có lớp cho trẻ em không?", "Dạ có ạ, bên em có các lớp Starters, Movers, Flyers cho bé từ 6-11 tuổi ạ."],
    
    # Policies & Quality
    ["Học IELTS ở đây có cam kết đầu ra không?", "Dạ có ạ. Trung tâm cam kết đầu ra bằng văn bản. Nếu không đạt điểm mục tiêu, anh/chị được học lại miễn phí ạ."],
    ["Lớp giao tiếp có giáo viên nước ngoài không?", "Dạ lớp Giao tiếp bên em học 100% với giáo viên bản ngữ (Anh/Mỹ/Úc) có chứng chỉ giảng dạy quốc tế ạ."],
    ["Một lớp tối đa bao nhiêu bạn?", "Dạ để đảm bảo chất lượng, mỗi lớp bên em chỉ từ 10-15 học viên thôi ạ."],
    ["Có được học thử không?", "Dạ anh/chị được đăng ký học thử miễn phí 2 buổi đầu tiên để trải nghiệm phương pháp dạy ạ."],
    
    # Enrollment & Payment
    ["Đăng ký rồi có được hoàn học phí không?", "Dạ nếu rút học phí trước ngày khai giảng 7 ngày thì được hoàn 100%. Sau đó thì bên em xin phép không hoàn lại ạ."],
    ["Có cho đóng học phí trả góp không?", "Dạ trung tâm có hỗ trợ trả góp 0% lãi suất qua thẻ tín dụng liên kết với 20 ngân hàng ạ."],
    ["Nếu bận có được bảo lưu không?", "Dạ anh/chị được bảo lưu tối đa 6 tháng nếu có lý do chính đáng (công tác, ốm đau...) ạ."],
    
    # Procedures / Tests
    ["Kiểm tra đầu vào có mất phí không?", "Dạ kiểm tra đầu vào (Placement Test) là hoàn toàn miễn phí ạ."],
    ["Bao lâu thì có lịch học mới?", "Dạ khóa mới khai giảng liên tục vào ngày 1 và 15 hàng tháng ạ."],
    ["Học xong có chứng chỉ không?", "Dạ cuối khóa mình sẽ có bài thi và được cấp chứng nhận hoàn thành khóa học của trung tâm ạ."],
    
    # Other
    ["Giáo trình bên mình dùng là gì?", "Dạ giáo trình do trung tâm biên soạn độc quyền kết hợp với các tài liệu chuẩn Cambridge/Oxford ạ."],
    ["Có lớp 1 kèm 1 không?", "Dạ có ạ, hình thức VIP 1-1 học phí sẽ cao hơn lớp thường, anh/chị có muốn em báo giá chi tiết không ạ?"],
    ["Xin chào", "Dạ chào anh/chị, em có thể giúp gì cho mình ạ?"]
]

df = pd.DataFrame(data, columns=["Question", "Answer"])

output_path = "faq_sample.xlsx"
df.to_excel(output_path, index=False, header=False) # No header as requested: Col A=Q, Col B=A

print(f"Created sample FAQ file at: {os.path.abspath(output_path)}")

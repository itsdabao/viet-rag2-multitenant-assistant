import os
import sys
import shutil

def fix_dlls_v2():
    # 1. Xác định nơi chứa (site-packages)
    env_path = sys.prefix
    site_packages = os.path.join(env_path, 'Lib', 'site-packages')
    
    # 2. Đích đến: Nơi thư viện llama_cpp đang nằm chờ (như trong ảnh bạn chụp)
    target_dir = os.path.join(site_packages, 'llama_cpp', 'lib')
    
    # 3. Nguồn: Folder 'nvidia' do lệnh pip vừa tạo ra
    nvidia_dir = os.path.join(site_packages, 'nvidia')
    
    print(f"📂 Đang tìm DLL trong: {nvidia_dir}")
    print(f"🎯 Đích đến: {target_dir}")
    
    if not os.path.exists(nvidia_dir):
        print("❌ LỖI: Không tìm thấy thư mục 'nvidia'. Có vẻ lệnh pip install chưa thành công?")
        return

    # Danh sách 3 file "thần thánh" cần tìm
    required_dlls = ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"]
    copied_count = 0

    # 4. Quét sâu trong thư mục nvidia
    for root, dirs, files in os.walk(nvidia_dir):
        for file in files:
            if file in required_dlls:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_dir, file)
                
                try:
                    shutil.copy2(src_file, dst_file)
                    print(f"✅ Đã tìm thấy và Copy: {file}")
                    copied_count += 1
                    # Đánh dấu là đã tìm thấy để không copy trùng
                    if file in required_dlls: required_dlls.remove(file) 
                except Exception as e:
                    print(f"⚠️ Lỗi khi copy {file}: {e}")

    # 5. Kết quả
    if copied_count >= 3 or len(required_dlls) == 0:
        print("\n🎉 THÀNH CÔNG RỰC RỠ!")
        print("👉 Bạn đã có đủ DLL. Hãy chạy lại model_test.py ngay!")
    else:
        print(f"\n⚠️ Vẫn thiếu {len(required_dlls)} file: {required_dlls}")

if __name__ == "__main__":
    fix_dlls_v2()
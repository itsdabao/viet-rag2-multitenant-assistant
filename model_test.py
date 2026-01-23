from llama_cpp import Llama

# 1. Khai báo đường dẫn tới file model bạn vừa tải
# (Đảm bảo tên file chính xác 100%)
MODEL_PATH = "models\qwen2.5-3b-instruct-q4_k_m.gguf"

print(">>> Đang khởi động AI trên GPU RTX 3050...")

# 2. Load Model
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,  # Số -1 nghĩa là: "Dùng hết sức mạnh GPU cho tao"
    n_ctx=2048,       # Bộ nhớ ngắn hạn (RAM)
    verbose=True      # Hiện thông số kỹ thuật (để bạn khoe với sếp)
)

# 3. Chat thử
question = "Chào bạn, hãy giới thiệu ngắn gọn về bản thân bằng tiếng Việt."
print(f"\nUser: {question}")

output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "Bạn là trợ lý AI hữu ích."},
        {"role": "user", "content": question}
    ],
    temperature=0.7,
)

print(f"AI: {output['choices'][0]['message']['content']}")
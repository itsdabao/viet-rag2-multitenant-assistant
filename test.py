from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq() # Tự động lấy key từ biến môi trường

completion = client.chat.completions.create(
    model="openai/gpt-oss-120b", # ID này đúng theo tài liệu bạn gửi
    messages=[
        {
            "role": "user",
            "content": "Explain why fast inference is critical for reasoning models"
        }
    ]
)

print(completion.choices[0].message.content)
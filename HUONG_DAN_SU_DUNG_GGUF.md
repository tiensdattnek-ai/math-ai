# Hướng dẫn dùng file GGUF sau khi train xong

Sau khi train trên Colab, bạn nhận được file (kích thước ước tính cho model 3B):
- `gia-su-toan-qwen2.5-3b-...-q4_k_m.gguf` — khoảng **2.0 GB** → dùng bản này là mặc định (cân bằng tốc độ/chất lượng)
- `...-q8_0.gguf` — khoảng **3.3 GB** → chất lượng cao hơn một chút, tốn RAM hơn

## Cách 1 — Ollama (dễ nhất, Windows/macOS/Linux)
1. Cài Ollama từ https://ollama.com
2. Mở file `Modelfile` trong project, sửa dòng `FROM ...` thành đường dẫn tới file `.gguf` của bạn
3. Mở terminal tại thư mục chứa GGUF + Modelfile:
   ```bash
   ollama create gia-su-toan -f Modelfile
   ollama run gia-su-toan
   ```
4. Thử hỏi: `Giải chi tiết: Tìm hai số biết tổng bằng 70, hiệu bằng 14` hoặc `Gợi ý cho em bài x² - 5x + 6 = 0, đừng cho đáp án ngay`

## Cách 2 — LM Studio (giao diện đẹp, hợp giáo viên/phụ huynh)
1. Cài LM Studio từ https://lmstudio.ai
2. File → Import Model (hoặc kéo thả file `.gguf` vào cửa sổ)
3. Trong ô System Prompt, dán nội dung SYSTEM ở cuối file `Modelfile` (phần trong ngoặc ba)
4. Nhiệt độ (temperature) để 0.3–0.6 cho Toán (thấp = ít lan man, Tính án ổn định hơn)

## Cách 3 — llama.cpp (máy yếu / muốn tùy chỉnh)
```bash
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
cmake -B build && cmake --build build -j
./build/bin/llama-cli -m /duong-dan/gia-su-toan-q4_k_m.gguf -cnv --ctx-size 4096
```

## Kiểm tra chất lượng model
Dán lần lượt 20 câu hỏi trong `data/test_prompts.md`, chấm theo checklist ở cuối file đó.
Nếu model hay sai ở một lớp/chủ đề: bổ sung thêm dữ liệu dạng đó (xem mục nâng cấp trong README) và train thêm 1 epoch.

## Lưu ý trung thực về giới hạn của model 3B
- Model 3B chạy offline trên máy cá nhân sẽ **không ngang trình** các AI lớn trên cloud ở bài Toán khó, chứng minh dài, đề phân loại cao.
- Điểm mạnh dự kiến: bài Tính toán theo dạng lớp 1–12, giải thích từng bước tiếng Việt, chế độ gợi ý làm bài tập về nhà.
- Luôn khuyên học sinh đối chiếu với lời giải của giáo viên/SGK với những bài quan trọng.

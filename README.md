# 🎓 AI Gia Sư Toán Việt Nam (Lớp 1 → Tốt nghiệp THPT) — Fine-tune Qwen2.5-3B & Xuất GGUF

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tiensdattnek-ai/math-ai/blob/main/Train_AI_GiaSu_Toan_Colab.ipynb)

> 👆 **Bấm nút "Open in Colab" ở trên** để mở thẳng notebook train trên Google Colab (nhớ chọn GPU T4 trong Runtime). Đây là cách bắt đầu nhanh nhất — không cần tải file về rồi upload lại.

Bộ công cụ hoàn chỉnh để bạn tự train một AI gia sư Toán tiếng Việt bằng **Google Colab GPU miễn phí**, rồi xuất thành **file GGUF** chạy offline trên máy của bạn (Ollama / LM Studio / llama.cpp).

## ⚠️ Trạng thái trung thực — đọc trước tiên

- Môi trường tạo project này **không có GPU** (2 CPU, ~2GB RAM) nên **chưa train tại đây và chưa có sẵn file `.gguf`** trong gói này. Bất kỳ ai khẳng định đã train được một LLM 3B tử tế trong môi trường như vậy đều không đáng tin.
- Gói này chứa **toàn bộ những gì cần để train thật**: dataset 6.000 mẫu đã sinh sẵn và kiểm chứng, notebook Colab chạy trọn từ cài đặt → train → test → xuất GGUF, scripts cho máy có GPU riêng, file cấu hình Ollama và bộ câu hỏi chấm điểm.
- Việc của bạn: mở notebook trên Colab có GPU T4, chạy theo các bước (bản thử 45–90 phút), tải file `.gguf` (~2,0 GB ở mức q4_k_m) về máy.

## 🚀 Bắt đầu trong 3 Bước

### Bước 1 — Mở notebook trên Colab
Bấm nút **Open in Colab** ở đầu trang này để mở thẳng `Train_AI_GiaSu_Toan_Colab.ipynb` trên Google Colab. (Cách dự phòng: tải file notebook về rồi upload thủ công tại [colab.research.google.com](https://colab.research.google.com) — File → Upload notebook.) Bạn **chỉ cần duy nhất file này** — bộ sinh dataset đã được nhúng sẵn bên trong.

### Bước 2 — Bật GPU và chạy từng Bước
`Runtime → Change runtime type → GPU T4`. Chạy lần lượt các ô. Lần đầu cứ giữ nguyên `CHE_DO = "THU_NHANH"` ở Bước ③ (1 epoch, ~6.000 mẫu) để có file GGUF thử nghiệm sớm. Hài lòng rồi mới đổi sang `"BAN_CHINH"` (trộn thêm ~8.000 mẫu Toán mở tiếng Việt, 2 epochs, ~2–4 giờ).

### Bước 3 — Tải GGUF về, chạy offline
Ở Bước ⑬, file GGUF được lưu Google Drive hoặc tải thẳng về máy. Sau đó làm theo **`HUONG_DAN_SU_DUNG_GGUF.md`** (Ollama với file `Modelfile` đính kèm là cách nhanh nhất), và chấm chất lượng bằng 20 câu trong `data/test_prompts.md`.

> 🛟 **Train xong nhưng bước xuất/tải GGUF báo không thấy file, trên Drive chỉ còn thư mục LoRA?** Đừng train lại. Mở ngay notebook cứu hộ **[CuuHo_Xuat_GGUF_Tu_LoRA_Colab.ipynb](https://colab.research.google.com/github/tiensdattnek-ai/math-ai/blob/main/CuuHo_Xuat_GGUF_Tu_LoRA_Colab.ipynb)** — nó tự tìm LoRA Adapter trên Drive, gộp vào model gốc và đóng gói GGUF thủ công bằng llama.cpp (15–30 phút). Phần train của bạn nằm trọn trong LoRA, đóng gói lỗi thì chỉ cần đóng gói lại.

## 🧠 Thiết kế model đã chốt với bạn

| Thành phần | Lựa chọn | Lý do |
|---|---|---|
| Base model | `unsloth/Qwen2.5-3B-Instruct` | Qwen2.5-Math không có bản 3B (chỉ 1.5B/7B); bản 3B-Instruct cân bằng tiếng Việt + Toán, GGUF q4 ~2GB chạy được laptop 8GB RAM |
| Phương pháp | QLoRA 4-bit (Unsloth), `r=16`, seq 2048 | Vừa GPU T4 16GB miễn phí của Colab |
| Dữ liệu chính | Tự sinh theo SGK VN lớp 1–12 (`data/train.jsonl`) | Đáp án do Python Tính (`math`/`Fraction`) → không học phải đáp án bịa |
| Dữ liệu bổ trợ (bản chính) | Orca-Math 200K + MetaMathQA 40K bản dịch tiếng Việt trên HuggingFace (lấy ngẫu nhiên ~8.000 mẫu) | Mở rộng dạng bài lời văn/suy luận ngoài các mẫu tự sinh |
| Phong cách | **2 chế độ** trong một model | Nói *"giải chi tiết"* → giải từng bước + đáp án + bài luyện; nói *"gợi ý"* → Socratic, không cho đáp án ngay, kết thúc bằng câu hỏi |
| Đầu ra | GGUF `q4_k_m` (mặc định ~2,0 GB), tuỳ chọn thêm `q8_0` (~3,3 GB) | Chuẩn của llama.cpp/Ollama/LM Studio |

Nguồn tham chiếu kỹ thuật: trang model Unsloth Qwen2.5-3B-Instruct trên Hugging Face; tài liệu *Saving to GGUF* của Unsloth (`save_pretrained_gguf(..., quantization_method="q4_k_m")`); các dataset `5CD-AI/Vietnamese-microsoft-orca-math-word-problems-200k-gg-translated` và `5CD-AI/Vietnamese-meta-math-MetaMathQA-40K-gg-translated` trên HuggingFace.

## 📊 Dataset đã sinh sẵn trong gói (`data/`)

- **6.000 mẫu**: 5.700 train + 300 validation · 3.841 giải chi tiết · 2.159 gợi ý
- Phân bố lớp 1–12: 510 · 448 · 237 · 278 · 758 · 436 · 395 · 678 · 503 · 484 · 563 · 710 mẫu (bộ sinh dùng 28 họ bài có tham số ngẫu nhiên; một số dạng tổ hợp số ít nên bão hoà sớm — xem `dataset_stats.json`)
- Đã kiểm tự động: đúng định dạng `messages` (system/user/assistant), không mẫu gợi ý nào lộ trường đáp án
- Sinh thêm bản lớn hơn bất cứ lúc nào: `python scripts/generate_dataset.py --n 20000 --out_dir data`
- Bản đồ chủ đề theo lớp và giới hạn bao phủ (phần chứng minh hình học/bài khó phân loại không tự sinh bằng mẫu được): `docs/chuong_trinh_lop1_12.md`

## 🗂️ Cấu trúc project

```
ai-gia-su-toan/
├── Train_AI_GiaSu_Toan_Colab.ipynb   ← BẮT ĐẦU TẠI ĐÂY (chỉ cần file này trên Colab)
├── README.md                          ← file này
├── HUONG_DAN_SU_DUNG_GGUF.md          ← cách chạy GGUF: Ollama / LM Studio / llama.cpp
├── Modelfile                          ← cấu hình sẵn cho Ollama (sửa 1 dòng FROM)
├── requirements.txt
├── config/training_config.json        ← toàn bộ siêu tham số ở một nơi
├── data/
│   ├── train.jsonl / val.jsonl        ← dataset đã sinh + kiểm chứng
│   ├── dataset_stats.json
│   └── test_prompts.md                ← 20 câu chấm model (tiểu học → THPT) + checklist
├── docs/chuong_trinh_lop1_12.md       ← bản đồ chương trình & cách mở rộng dữ liệu
└── scripts/
    ├── generate_dataset.py            ← bộ sinh dữ liệu SGK (chạy được bằng CPU)
    ├── load_open_datasets.py          ← tải & chuẩn hoá dataset mở tiếng Việt
    ├── train_lora.py                  ← bản script của bước train (máy có GPU riêng)
    ├── export_gguf.py                 ← bản script của bước xuất GGUF (+ cách thủ công dự phòng)
    └── build_notebook.py              ← dựng lại notebook Colab từ các script trên
```

## 💻 Nếu bạn có GPU riêng / thuê cloud thay vì Colab

```bash
pip install -r requirements.txt
python scripts/generate_dataset.py --n 8000 --out_dir data           # nếu muốn sinh mới
python scripts/load_open_datasets.py --orca_n 4000 --meta_n 4000 --out data/open.jsonl
python scripts/train_lora.py --extra_data data/open.jsonl --epochs 2 --output_dir outputs/lora
python scripts/export_gguf.py --adapter_dir outputs/lora --out_dir outputs/gguf --quant q4_k_m
```

## 📈 Lộ trình nâng cấp chất lượng (sau bản đầu)

1. **Chấm bằng dữ kiện, không chấm bằng cảm giác:** tạo thêm 100–200 bài "đề thi thử" không nằm trong dữ liệu train, chấm tỉ lệ đáp án cuối đúng theo từng lớp/chủ đề bằng checklist trong `test_prompts.md`.
2. **Thêm dữ liệu giáo viên soạn** cho dạng còn yếu (đặc biệt chứng minh hình học, bài thực tế lớp 9/12) — định dạng giống `data/train.jsonl`, 500 bài chất lượng cao tốt hơn 50.000 bài dịch máy nhiễu.
3. **Train bản lớn hơn khi cần cấp 3:** đổi base sang Qwen2.5-Math-7B trên GPU thuê (A100/L4), giữ nguyên pipeline và dataset; GGUF q4 khi đó ~4–5 GB.
4. **An toàn sư phạm:** model 3B vẫn có thể sai ở bài khó dù trình bày tự tin. Khi dùng cho học sinh, hãy coi đây là trợ lý luyện tập và đối chiếu bài quan trọng với giáo viên/SGK.

---
*Tạo ngày 20/10/2026 theo cấu hình đã chốt: Colab GPU miễn phí · Qwen2.5-3B · dữ liệu SGK VN + dataset mở · hai chế độ Giải chi tiết / Gợi ý.*

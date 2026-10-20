#!/usr/bin/env python3
"""Dựng file Train_AI_GiaSu_Toan_Colab.ipynb từ các thành phần trong project.
Bộ sinh dataset được mã hóa base64 nhúng vào notebook để chỉ cần 1 file .ipynb vẫn chạy trọn trên Colab.
Chạy: python scripts/build_notebook.py
"""
import base64, json, os, textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def b64_chunks(path, width=110):
    raw = open(path, "rb").read()
    return "\n".join(textwrap.wrap(base64.b64encode(raw).decode(), width))

gen_b64 = b64_chunks(os.path.join(ROOT, "scripts", "generate_dataset.py"))
open_b64 = b64_chunks(os.path.join(ROOT, "scripts", "load_open_datasets.py"))

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}
def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.strip("\n").splitlines(True)}

cells = []

cells.append(md(r"""
# 🎓 Train AI Gia Sư Toán Việt Nam — Lớp 1 → Tốt nghiệp THPT → Xuất GGUF

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tiensdattnek-ai/math-ai/blob/main/Train_AI_GiaSu_Toan_Colab.ipynb)

**Base model:** `unsloth/Qwen2.5-3B-Instruct` · **Phương pháp:** QLoRA bằng Unsloth · **Kết quả cuối:** file `.gguf` chạy offline trên Ollama / LM Studio / llama.cpp

### Trước khi bắt đầu (bắt buộc)
1. Trên Google Colab: **Thời gian chạy (Runtime) → Thay đổi loại thời gian chạy → chọn GPU T4** (miễn phí). Không có GPU thì không train được.
2. Chạy lần lượt từ trên xuống (**không** nên bấm Run all ngay lần đầu — hãy đọc ghi chú của từng bước).
3. Ở **Bước 3** có công tắc `CHE_DO`: để `"THU_NHANH"` cho lần chạy đầu (khoảng 45–90 phút, ra GGUF thử được ngay), xong mới đổi sang `"BAN_CHINH"` (2–4 giờ, trộn thêm dataset mở tiếng Việt) khi muốn bản tốt hơn.

> **Kỳ vọng thực tế:** Fine-tune này dạy model phong cách gia sư tiếng Việt (giải từng bước + chế độ gợi ý Socratic không lộ đáp án) trên nền kiến thức Toán Qwen2.5 đã có. Model 3B offline sẽ giỏi các dạng Tính toán theo mẫu lớp 1–12, nhưng không thay thế được giáo viên/AI cloud lớn ở bài chứng minh khó. Hãy chấm model bằng 20 câu kiểm tra kèm trong project (`data/test_prompts.md`).

**Kế hoạch notebook:** ① Kiểm tra GPU → ② Cài thư viện → ③ Cấu hình → ④ (Tuỳ chọn) gắn Google Drive → ⑤ Sinh dataset SGK VN tự động → ⑥ Nạp dataset mở → ⑦ Xem dữ liệu → ⑧ Nạp model + LoRA → ⑨ Định dạng dữ liệu → ⑩ Train → ⑪ Test thử → ⑫ Xuất GGUF → ⑬ Tải file về / lưu Drive → ⑭ (Tuỳ chọn) test GGUF.
"""))

cells.append(md("## Bước ①: Kiểm tra GPU"))
cells.append(code(r"""
!nvidia-smi
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name} | VRAM: {p.total_memory/1e9:.1f} GB | bf16 ho tro: {torch.cuda.is_bf16_supported()}")
else:
    print("KHONG CO GPU! Hay vao Runtime -> Change runtime type -> GPU T4 roi chay lai.")
"""))

cells.append(md(r"""
## Bước ②: Cài đặt thư viện
Chạy xong ô này, nếu Colab báo cần khởi động lại phiên (*Restart session*) thì bấm đồng ý, rồi chạy tiếp từ Bước ③ (không cần chạy lại ô cài đặt).
"""))
cells.append(code(r"""
%%capture
# Cai Unsloth + cac thu vien train. Neu loi, xem lenh Colab moi nhat tai docs.unsloth.ai (muc Installing)
!pip install -q unsloth datasets trl peft accelerate bitsandbytes sentencepiece protobuf
print("Cai dat xong.")
"""))

cells.append(md(r"""
## Bước ③: Cấu hình — CHỈ CẦN SỬA Ô NÀY
- `THU_NHANH`: chỉ dùng 6.000 mẫu SGK tự sinh, 1 epoch → mục đích là kiểm chứng toàn bộ đường ống tới khi có file GGUF trong thời gian ngắn nhất.
- `BAN_CHINH`: thêm ~8.000 mẫu dataset mở tiếng Việt (Orca-Math + MetaMath đã dịch), 2 epochs → model tốt hơn nhưng dễ chạm giới hạn phiên Colab free, nên bật lưu Google Drive ở bước ④.
"""))
cells.append(code(r"""
CHE_DO = "THU_NHANH"          # doi thanh "BAN_CHINH" khi ban da chay thu thanh cong
LUU_LEN_DRIVE = True          # True: moc Google Drive de luu GGUF/checkpoint ben vung

if CHE_DO == "THU_NHANH":
    SO_MAU_TU_SINH, EPOCHS, DUNG_OPEN_DATASET = 6000, 1, False
else:
    SO_MAU_TU_SINH, EPOCHS, DUNG_OPEN_DATASET = 8000, 2, True

BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct"   # Qwen2.5-Math khong co ban 3B; ban 3B Instruct nay can bang tieng Viet + Toan
MOUNTED_DRIVE_DIR = "/content/drive/MyDrive/ai-gia-su-toan"
LOCAL_OUT = "/content/outputs"
LORA_DIR = f"{LOCAL_OUT}/lora"
GGUF_DIR = f"{LOCAL_OUT}/gguf"
MAX_SEQ = 2048
QUANTS = ["q4_k_m"]           # them "q8_0" neu muon ban GGUF ~3.3GB chat luong cao hơn
print(CHE_DO, "| mau tu sinh:", SO_MAU_TU_SINH, "| epochs:", EPOCHS, "| open dataset:", DUNG_OPEN_DATASET)
"""))

cells.append(md("## Bước ④ (tuỳ chọn): Gắn Google Drive để không mất file khi Colab ngắt phiên"))
cells.append(code(r"""
import os
drive_mounted = False
if LUU_LEN_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        os.makedirs(MOUNTED_DRIVE_DIR, exist_ok=True)
        drive_mounted = True
        print("Da gan Drive:", MOUNTED_DRIVE_DIR)
    except Exception as e:
        print("Khong gan duoc Drive (vẫn train tiep duoc, nho tai GGUF ve ngay sau khi xong):", e)
os.makedirs(LOCAL_OUT, exist_ok=True)
"""))

cells.append(md(r"""
## Bước ⑤: Sinh dataset Toán SGK Việt Nam lớp 1–12 (tự động)
Ô này giải mã bộ sinh dữ liệu đã nhúng sẵn, rồi sinh dữ liệu: 28 họ bài toán xuyên suốt lớp 1–12, đáp án do Python Tính xác nên không có đáp án bịa. Mỗi bài có một trong hai chế độ: **Giải chi tiết** (từng bước + đáp án + bài luyện) hoặc **Gợi ý Socratic** (không lộ đáp án, kết thúc bằng câu hỏi cho học sinh).

*Nếu bạn đã tải project lên và có sẵn `data/train.jsonl` trong thư mục làm việc, ô này sẽ dùng file đó thay vì sinh lại.*
"""))
cells.append(code(f"""
import base64, os
os.makedirs("scripts", exist_ok=True)
GEN_B64 = \"\"\"{gen_b64}\"\"\"
OPEN_B64 = \"\"\"{open_b64}\"\"\"
open("scripts/generate_dataset.py","wb").write(base64.b64decode(GEN_B64))
open("scripts/load_open_datasets.py","wb").write(base64.b64decode(OPEN_B64))
print("Da khoi phuc scripts:", os.listdir("scripts"))
"""))
cells.append(code(r"""
import os
if os.path.exists("data/train.jsonl"):
    print("Tim thay data/train.jsonl co san -> dung file nay, bo qua sinh lai.")
else:
    !python scripts/generate_dataset.py --n {SO_MAU_TU_SINH} --out_dir data
""".replace("{SO_MAU_TU_SINH}", "6000 if False else 0") if False else r"""
import subprocess, os, sys
if os.path.exists("data/train.jsonl"):
    print("Tim thay data/train.jsonl co san -> dung file nay, bo qua sinh lai.")
else:
    cmd = [sys.executable, "scripts/generate_dataset.py", "--n", str(SO_MAU_TU_SINH), "--out_dir", "data"]
    print("Chạy:", " ".join(cmd))
    subprocess.run(cmd, check=True)
"""))

cells.append(md(r"""
## Bước ⑥: Nạp dataset Toán mở tiếng Việt (chỉ chạy ở chế độ BAN_CHINH)
Nguồn: `5CD-AI/Vietnamese-microsoft-orca-math-word-problems-200k-gg-translated` và `5CD-AI/Vietnamese-meta-math-MetaMathQA-40K-gg-translated` trên HuggingFace. Script tự dò tên cột, nếu một nguồn Lỗi sẽ bỏ qua nguồn đó mà không làm dừng notebook. Chất lượng dịch máy của các tập này không hoàn hảo — đây là lý do dữ liệu SGK tự sinh ở bước ⑤ vẫn là trục chính.
"""))
cells.append(code(r"""
import subprocess, sys, os
if DUNG_OPEN_DATASET:
    r = subprocess.run([sys.executable, "scripts/load_open_datasets.py", "--orca_n", "4000", "--meta_n", "4000", "--out", "data/open.jsonl"])
    print("Xong, ma tra ve:", r.returncode, "| file ton tai:", os.path.exists("data/open.jsonl"))
else:
    print("Bo qua (che do THU_NHANH).")
"""))

cells.append(md("## Bước ⑦: Kiểm tra dữ liệu trước khi train"))
cells.append(code(r"""
import json, random
from collections import Counter

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

rows = load_jsonl("data/train.jsonl")
if DUNG_OPEN_DATASET and os.path.exists("data/open.jsonl"):
    rows += load_jsonl("data/open.jsonl")
random.Random(42).shuffle(rows)
print("Tong so mau train:", len(rows))
print("Theo Lop:", dict(sorted(Counter(r.get("grade",0) for r in rows).items())))
print("Theo che do:", dict(Counter(r.get("mode","?") for r in rows)))
for r in rows[:2]:
    print("\nUSER:\n", r["messages"][1]["content"])
    print("ASSISTANT:\n", r["messages"][2]["content"])
    print("-" * 70)
"""))

cells.append(md(r"""
## Bước ⑧: Nạp Qwen2.5-3B-Instruct + gắn LoRA
`load_in_4bit=True` (QLoRA) giúp model 3B vừa GPU T4 16GB. Chỉ một phần nhỏ trọng số LoRA được tập — vì vậy bước xuất GGUF ở cuối sẽ **gộp** LoRA vào model gốc trước khi lượng tử hoá. Nếu bị *out of memory* ở bước train: giảm `batch_size` xuống 1 và `MAX_SEQ` xuống 1024 trong bước ③/⑩.
"""))
cells.append(code(r"""
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ,
    dtype=None,          # tu chon bf16/fp16
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=32, lora_dropout=0.05, bias="none",
    use_gradient_checkpointing="unsloth", random_state=42, use_rslora=True,
)
print("San sang train. Device:", next(model.parameters()).device)
"""))

cells.append(md("## Bước ⑨: Định dạng dữ liệu bằng chat template của Qwen (ChatML)"))
cells.append(code(r"""
from datasets import Dataset

def to_text(row):
    text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
    if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
        text += tokenizer.eos_token
    return {"text": text}

train_dataset = Dataset.from_list([to_text(r) for r in rows])
print(train_dataset)
print(train_dataset[0]["text"][:800])
"""))

cells.append(md(r"""
## Bước ⑩: TRAIN 🚀
Theo dõi đồ thị **loss**: kỳ vọng loss giảm dần và ổn định. Loss không giảm/giảm rất chậm thường do learning rate quá thấp hoặc dữ liệu nhiễu; loss tụt về gần 0 quá nhanh ở dataset nhỏ là dấu hiệu học thuộc (overfit) — với bản chính nên dùng tập val để kiểm chứng. Checkpoint được lưu theo từng epoch.
"""))
cells.append(code(r"""
import inspect, torch
from trl import SFTTrainer, SFTConfig

# Xay dung cau hinh tuong thich nhieu phien ban TRL (thu vien nay doi API kha thuong xuyen)
sft_fields = set(inspect.signature(SFTConfig).parameters) if hasattr(inspect.signature(SFTConfig), "parameters") else set()
try:
    sft_kwargs = dict(
        output_dir=LORA_DIR, per_device_train_batch_size=2, gradient_accumulation_steps=8,
        num_train_epochs=EPOCHS, learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
        weight_decay=0.01, logging_steps=10, save_strategy="epoch", optim="adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
        report_to="none", packing=False,
    )
    try:
        cfg = SFTConfig(**sft_kwargs, dataset_text_field="text", max_length=MAX_SEQ)
    except TypeError:
        cfg = SFTConfig(**sft_kwargs, dataset_text_field="text", max_seq_length=MAX_SEQ)
except TypeError:
    cfg = SFTConfig(**sft_kwargs)

trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters)
t_kwargs = {"model": model, "train_dataset": train_dataset, "args": cfg}
t_kwargs["processing_class" if "processing_class" in trainer_params else "tokenizer"] = tokenizer
trainer = SFTTrainer(**t_kwargs)

trainer.train()
model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print("Da luu LoRA:", LORA_DIR)
if drive_mounted:
    import shutil
    dst = os.path.join(MOUNTED_DRIVE_DIR, "lora")
    if os.path.exists(dst): shutil.rmtree(dst)
    shutil.copytree(LORA_DIR, dst)
    print("Da sao luu LoRA len Drive:", dst)
"""))

cells.append(md(r"""
## Bước ⑪: Test thử ngay trên Colab (3 trình độ, 2 chế độ)
Đọc kỹ 3 câu trả lời. Câu "gợi ý" đạt yêu cầu chỉ khi nó **không** cho đáp án cuối cùng và kết thúc bằng một câu hỏi. Nếu câu trả lời cứ sinh tiếp không dừng, đừng lo — kiểm tra lại ở bản GGUF; nếu vẫn vậy, xem mục xử lý sự cố cuối notebook (thiếu EOS).
"""))
cells.append(code(r"""
from transformers import TextStreamer
FastLanguageModel.for_inference(model)
SYSTEM_PROMPT = rows[0]["messages"][0]["content"]
CASES = [
    "Giải chi tiết: Một xe máy đi 135 km trong 3 giờ. Tính vận tốc của xe máy.",
    "Gợi ý cho em bài này, đừng cho đáp án ngay: Giải phương trình x² - 7x + 12 = 0",
    "Giải chi tiết: Tìm cực trị của hàm số y = x³ - 3x.",
]
streamer = TextStreamer(tokenizer, skip_prompt=True)
for q in CASES:
    print("\n" + "=" * 30, "CAU HOI:", q)
    messages = [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":q}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    model.generate(input_ids=inputs, streamer=streamer, max_new_tokens=700, temperature=0.6, top_p=0.9, do_sample=True)
"""))

cells.append(md(r"""
## Bước ⑫: Xuất GGUF (q4_k_m) 🎯
Đây là bước tạo sản phẩm cuối. Unsloth sẽ gộp (merge) LoRA vào model gốc, chuyển sang GGUF (dùng llama.cpp bên dưới) và lượng tử hoá. Với model 3B: **q4_k_m ≈ 2.0 GB** (nên dùng), **q8_0 ≈ 3.3 GB**. Bước này có thể mất 5–20 phút và cần khá nhiều RAM/dung lượng đĩa tạm.

> ⚠️ **Đã ghi nhận trên thực tế:** hàm xuất GGUF của Unsloth có thể đặt file vào thư mục con/anh em hoặc lỗi ở khâu biên dịch llama.cpp/nén. Ô dưới đã vá bằng **quét đệ quy** mọi file `.gguf` trên 500 MB. Nếu vẫn trống, **đừng train lại** — mở notebook cứu hộ `CuuHo_Xuat_GGUF_Tu_LoRA_Colab.ipynb` (cùng repo) để xuất thủ công từ LoRA đã lưu.
"""))
cells.append(code(r"""
import glob, os, shutil, traceback
os.makedirs(GGUF_DIR, exist_ok=True)
export_errors = []
for q in QUANTS:
    print("Dang xuat GGUF:", q)
    try:
        model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method=q)
    except Exception as e:
        export_errors.append(f"{q}: {e}")
        traceback.print_exc()
        print("Luu ý: Loi o day KHONG lam mat cong train - LoRA van an toan o", LORA_DIR)

def collect_gguf(min_size=500_000_000):
    seen, out = set(), []
    for cur, dirs, files in os.walk("/content"):
        dirs[:] = [d for d in dirs if d not in ("drive", "sample_data", ".cache")]
        for name in files:
            if name.lower().endswith(".gguf"):
                p = os.path.join(cur, name)
                try:
                    rp = os.path.realpath(p)
                    if rp not in seen and os.path.getsize(p) >= min_size:
                        seen.add(rp); out.append(p)
                except OSError:
                    pass
    return sorted(out, key=lambda p: os.path.getsize(p))

found = collect_gguf()
print(f"\nTim thay {len(found)} file GGUF tren 500MB trong toan bo /content:")
normalized = []
for f in found:
    print(f" {os.path.getsize(f)/1e9:6.2f} GB  {f}")
    dst = os.path.join(GGUF_DIR, os.path.basename(f))
    if os.path.abspath(f) != os.path.abspath(dst):
        shutil.copy(f, dst)
    normalized.append(dst)
if not normalized:
    print("\n!!! CHUA CO FILE GGUF NAO. DUNG TRAIN LAI !!!")
    print("LoRA (toan bo cong train) van o:", LORA_DIR, "va ban sao len Drive tu Buoc 10 neu da bat.")
    print("Mo notebook CuuHo_Xuat_GGUF_Tu_LoRA_Colab.ipynb trong cung repo de xuat GGUF thu cong tu LoRA (15-30 phut).")
    if export_errors:
        print("Loi tu Unsloth:", *export_errors, sep="\n - ")
else:
    print("\nDa gom GGUF ve:", GGUF_DIR)
"""))

cells.append(md("## Bước ⑬: Lưu/Tải file GGUF (hãy làm NGAY kẻo Colab ngắt phiên mất file)"))
cells.append(code(r"""
import glob, shutil, os
files = sorted(set(
    glob.glob(f"{GGUF_DIR}/**/*.gguf", recursive=True) +
    glob.glob(f"{GGUF_DIR}_gguf/**/*.gguf", recursive=True)
))
files = [f for f in files if os.path.getsize(f) > 500_000_000]
if not files:
    raise RuntimeError(
        "Khong tim thay GGUF nao tren 500MB sau Buoc 12. DUNG TRAIN LAI: LoRA cua ban van an toan. "
        "Hay mo notebook CuuHo_Xuat_GGUF_Tu_LoRA_Colab.ipynb (cung repo) de xuat GGUF thu cong tu LoRA tren Drive.")
print("Cac file GGUF se luu/tai:")
for f in files: print(f" {os.path.getsize(f)/1e9:.2f} GB  {f}")
if drive_mounted:
    dst_dir = os.path.join(MOUNTED_DRIVE_DIR, "gguf")
    os.makedirs(dst_dir, exist_ok=True)
    for f in files:
        shutil.copy(f, dst_dir)
    print("Da luu GGUF len Google Drive:", dst_dir)
else:
    from google.colab import files as colab_files
    for f in files:
        print("Dang tai ve may:", f)
        colab_files.download(f)
"""))

cells.append(md(r"""
## Bước ⑭ (tuỳ chọn): Test file GGUF bằng llama-cpp-python
Ô này **không bắt buộc** — cài đặt mất vài phút và đôi khi Lỗi biên dịch trên Colab. Cách test đáng tin nhất là tải GGUF về máy rồi chạy Ollama/LM Studio theo `HUONG_DAN_SU_DUNG_GGUF.md` trong project. Chỉ chạy ô này nếu bạn muốn kiểm tra nhanh file GGUF có mở được không trước khi tải về.
"""))
cells.append(code(r"""
!pip install -q llama-cpp-python
import glob
from llama_cpp import Llama
path = sorted(glob.glob(f"{GGUF_DIR}/**/*.gguf", recursive=True))[0]  # tim de quy, lay file GGUF dau tien tim thay
llm = Llama(model_path=path, n_ctx=4096, n_gpu_layers=-1, verbose=False)
out = llm.create_chat_completion(messages=[
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Giải chi tiết: Tính diện tích hình chữ nhật dài 9 cm, rộng 6 cm."},
], max_tokens=500, temperature=0.6)
print(out["choices"][0]["message"]["content"])
"""))

cells.append(md(r"""
# ✅ Sau khi có file GGUF — làm gì tiếp?
1. **Ollama (khuyên dùng):** sửa dòng `FROM` trong file `Modelfile` của project thành đúng tên file `.gguf`, chạy `ollama create gia-su-toan -f Modelfile` rồi `ollama run gia-su-toan`.
2. **LM Studio:** import file `.gguf`, dán System Prompt (có trong Modelfile), để temperature 0.3–0.6 cho môn Toán.
3. **Chấm chất lượng:** hỏi đủ 20 câu trong `data/test_prompts.md` của project, ghi lại dạng bài nào còn sai → bổ sung dữ liệu dạng đó → train thêm.

## 🛠️ Xử lý sự cố thường gặp
| Triệu chứng | Nguyên nhân thường gặp | Cách sửa |
|---|---|---|
| `CUDA out of memory` khi train | batch/seq quá lớn cho VRAM | Giảm `per_device_train_batch_size` còn 1, `MAX_SEQ` còn 1024, tăng `gradient_accumulation_steps` lên 16 |
| Colab ngắt phiên giữa chừng | Hết thời lượng free | Bật `LUU_LEN_DRIVE=True`; train tiếp từ checkpoint: nạp `model_name` là folder LoRA/checkpoint đã lưu trên Drive |
| Model trả lời tiếng Anh hoặc lạc đề | Ít epoch/dữ liệu mở nhiễu | Chạy `BAN_CHINH` 2 epochs, hoặc tăng tỉ lệ dữ liệu tự sinh (sinh `--n 15000`) |
| Model sinh không dừng, lặp | Thiếu token kết thúc (EOS) | Kiểm tra bước ⑨ đã gắn EOS; khi chạy Ollama thêm stop `<|im_end|>` (Modelfile đã có sẵn) |
| `save_pretrained_gguf` báo lỗi / bước tải báo không thấy GGUF | Hàm xuất của Unsloth đặt file lạc thư mục hoặc lỗi biên dịch llama.cpp/khâu nén | **ĐỪNG TRAIN LẠI.** Bước ⑫ đã quét đệ quy toàn bộ `/content`; nếu vẫn trống, mở notebook `CuuHo_Xuat_GGUF_Tu_LoRA_Colab.ipynb` cùng repo — tự tìm LoRA trên Drive, gộp và convert thủ công (15–30 phút) |
| Chế độ gợi ý vẫn lộ đáp án | Model chưa học chắc ranh giới hai chế độ | Sinh thêm dữ liệu (tỉ lệ gợi ý tăng), nhấn mạnh trong prompt kiểm tra: *"đừng cho đáp án ngay"* |
| Đáp án đúng nhưng lý giải sai công thức | Giới hạn 3B ở bài khó | Đây là Lỗi cần loại khi chấm bằng test_prompts; bài khó nên để AI cloud/giáo viên xử lý |

## 🚀 Muốn model tốt hơn nữa?
- Tăng dữ liệu: thêm họ bài vào bộ sinh, hoặc đưa thêm dữ liệu do giáo viên soạn (đề + lời giải đúng chuẩn) — chất lượng quan trọng hơn số lượng thô.
- Thuê GPU mạnh hơn để train **Qwen2.5-Math-7B** nếu đối tượng chính là cấp 3/ôn thi tốt nghiệp (đổi `BASE_MODEL`, giảm batch size).
- Đánh giá định lượng: giữ riêng 100–200 bài kiểm tra không nằm trong dữ liệu train, chấm "đáp án cuối đúng" trước/sau mỗi lần tinh chỉnh.

**Chúc bạn train thành công và có một trợ lý gia sư hữu ích, trung thực về giới hạn của nó!**
"""))

nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    },
    "cells": cells,
}
out_path = os.path.join(ROOT, "Train_AI_GiaSu_Toan_Colab.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Đã tạo", out_path, "số ô:", len(cells))

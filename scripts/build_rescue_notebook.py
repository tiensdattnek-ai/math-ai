#!/usr/bin/env python3
"""Dựng notebook cứu hộ CuuHo_Xuat_GGUF_Tu_LoRA_Colab.ipynb.
Notebook nay KHÔNG train lai: nap LoRA adapter da luu tren Google Drive,
gop (merge) vao base model bang Transformers+PEFT chuan HuggingFace,
roi convert sang GGUF thu cong bang llama.cpp roi quantize Q4_K_M.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}
def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.strip("\n").splitlines(True)}

cells = []

cells.append(md(r"""
# 🛟 CỨU HỘ: Xuất GGUF từ LoRA đã train — KHÔNG cần train lại

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tiensdattnek-ai/math-ai/blob/main/CuuHo_Xuat_GGUF_Tu_LoRA_Colab.ipynb)

**Dùng notebook này khi:** bạn đã train xong ở notebook chính, trên Google Drive đã có thư mục **LoRA**, nhưng bước xuất/tải GGUF báo không thấy file.

**Vì sao không phải train lại?** LoRA adapter chính là toàn bộ "công sức train". Model gốc là file công khai tải lại được bất cứ lúc nào. File GGUF chỉ là kết quả **gộp Adapter + model gốc rồi đóng gói** — đóng gói lỗi thì đóng gói lại, khoảng **15–30 phút** trên T4.

**Notebook này làm khác notebook chính ở điểm nào?** Nó *không* dùng hàm xuất GGUF của Unsloth (bước đã gây lỗi). Thay vào đó: gộp bằng Transformers + PEFT chuẩn → convert thủ công bằng `llama.cpp/convert_hf_to_gguf.py` → nén `Q4_K_M` → kiểm tra file → chép lên Drive. Đường thủ công này chậm hơn vài phút nhưng từng bước đều nhìn thấy và kiểm chứng được.

### Chuẩn bị
1. **Runtime → Change runtime type → GPU T4.**
2. Nếu tab Colab cũ (buổi train) vẫn còn chạy, hãy ngắt runtime đó (Runtime → Disconnect and delete runtime) để giải phóng GPU — LoRA trên Drive là đủ, không phụ thuộc phiên cũ nữa.
3. Chạy lần lượt từ trên xuống. Dấu hiệu thành công ở cuối: Drive thư mục `ai-gia-su-toan/gguf/` có file `gia-su-toan-qwen2.5-3b-q4_k_m.gguf` nặng khoảng **1,9–2,2 GB**.
"""))

cells.append(md("## Bước 1: Kiểm tra GPU + đĩa"))
cells.append(code(r"""
import os, shutil, torch
print("CUDA:", torch.cuda.is_available(), "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "KHONG CO - hay bat T4 o Runtime!")
free_gb = shutil.disk_usage("/content").free / 1e9
print(f"Dung luong trong /content: {free_gb:.1f} GB (can tam 25GB tro len cho cac file trung gian)")
assert torch.cuda.is_available(), "Chua co GPU. Runtime -> Change runtime type -> GPU T4, roi chay lai tu dau."
"""))

cells.append(md("## Bước 2: Cấu hình (thường không cần sửa)"))
cells.append(code(r"""
DRIVE_PROJECT = "/content/drive/MyDrive/ai-gia-su-toan"   # noi notebook chinh da luu LoRA
BASE_FALLBACK = "Qwen/Qwen2.5-3B-Instruct"
MERGED_DIR    = "/content/merged_hf"
F16_GGUF      = "/content/gia-su-toan-f16.gguf"
FINAL_GGUF    = "/content/gia-su-toan-qwen2.5-3b-q4_k_m.gguf"
QUANT         = "Q4_K_M"
TAI_THANG_VE_MAY = False   # True neu muon trinh duyet tai file 2GB ve may ngay tai buoc cuoi (Drive la backups)
print("Se tim LoRA trong:", DRIVE_PROJECT)
"""))

cells.append(md("## Bước 3: Gắn Google Drive"))
cells.append(code(r"""
from google.colab import drive
drive.mount("/content/drive")
import os
print("Thu muc project ton tai tren Drive:", os.path.exists(DRIVE_PROJECT))
if os.path.exists(DRIVE_PROJECT):
    for n in sorted(os.listdir(DRIVE_PROJECT)):
        print(" -", n)
"""))

cells.append(md(r"""
## Bước 4: Kiểm tra xem GGUF có sẵn nhưng bị nằm "lạc chỗ" không
Bước xuất GGUF của Unsloth có khi đặt file vào thư mục con hoặc tạo xong file 16-bit trung gian rồi mới lỗi ở khâu nén. Ô này quét đệ quy `/content` (bỏ qua thư mục Drive) và thư mục project trên Drive:
- Thấy file đã nén (~2 GB, tên có Q4) → các bước nặng phía dưới tự bỏ qua, đi thẳng đến bước chép file cuối.
- Chỉ thấy file F16/BF16 (~6 GB) → bỏ qua đoạn gộp/convert, chỉ việc nén tiếp ở Bước 9.
- Không thấy gì → làm đầy đủ các bước từ LoRA.
"""))
cells.append(code(r"""
import os

def find_files(root, suffix, min_size=0, skip=("sample_data",), max_depth=12):
    out = []
    if not os.path.exists(root): return out
    base_depth = root.rstrip("/").count("/")
    for cur, dirs, files in os.walk(root):
        if cur.count("/") - base_depth > max_depth: dirs[:] = []; continue
        dirs[:] = [d for d in dirs if d not in skip]
        for f in files:
            if f.lower().endswith(suffix):
                p = os.path.join(cur, f)
                try:
                    if os.path.getsize(p) >= min_size: out.append(p)
                except OSError: pass
    return out

found = find_files("/content", ".gguf", min_size=500_000_000, skip=("drive", "sample_data"))
found += [f for f in find_files(DRIVE_PROJECT, ".gguf", min_size=500_000_000) if f not in found]
found.sort(key=lambda p: os.path.getsize(p))
for p in found:
    print(f"{os.path.getsize(p)/1e9:6.2f} GB  {p}")

EXISTING_Q   = next((p for p in found if "q4" in os.path.basename(p).lower()), None)
EXISTING_F16 = None if EXISTING_Q else next((p for p in found if any(t in os.path.basename(p).lower() for t in ("f16","bf16")) or os.path.getsize(p) > 4e9), None)
print("\nGGUF da nen san:", EXISTING_Q)
print("GGUF 16-bit trung gian:", EXISTING_F16)
if EXISTING_Q:   print(">> Tin vui: file da co san! Cac buoc nang se tu bo qua, cu chay het xuong buoc cuoi de chep len Drive.")
elif EXISTING_F16: print(">> Co file 16-bit: se bo qua doan gop/convert, chi can nen tiep o Buoc 9.")
else: print(">> Chua co GGUF nao: lam day du tu LoRA o cac buoc sau (van khong phai train lai).")
"""))

cells.append(md(r"""
## Bước 5: Tự động tìm LoRA Adapter trên Drive
Adapter được nhận diện bằng file `adapter_config.json` kèm trọng số `adapter_model.safetensors`. Notebook chính lưu nó ở `.../ai-gia-su-toan/lora/`. Nếu chỉ thấy các thư mục `checkpoint-...` (do phiên bị ngắt), ô này tự chọn checkpoint mới nhất — được ưu tiên đúng thứ tự: thư mục tên `lora` → checkpoint số lớn nhất → file mới nhất.
"""))
cells.append(code(r"""
import os, re, json, shutil

if EXISTING_Q or EXISTING_F16:
    print("Buoc nay duoc bo qua vi da tim thay GGUF o Buoc 4.")
    LORA_SRC = None
else:
    configs = find_files(DRIVE_PROJECT, "adapter_config.json")
    if not configs:
        print("Chua thay trong", DRIVE_PROJECT, "- dang quet toan bo MyDrive (co the mat vai phut)...")
        configs = find_files("/content/drive/MyDrive", "adapter_config.json", max_depth=8)
    assert configs, ("KHONG tim thay adapter_config.json tren Drive. Hay mo Google Drive kiem tra ban thay thu muc LoRA "
                      "o duong dan nao, va ben trong co file adapter_config.json + adapter_model.safetensors khong, "
                      "roi sua DRIVE_PROJECT o Buoc 2 / gui ten thu muc do de duoc ho tro.")

    def score(cfg_path):
        d = os.path.dirname(cfg_path); base = os.path.basename(d)
        m = re.search(r"checkpoint-(\d+)", d)
        has_w = any(os.path.exists(os.path.join(d, n)) for n in ("adapter_model.safetensors", "adapter_model.bin"))
        return (has_w, 2 if base == "lora" else 0, int(m.group(1)) if m else -1, os.path.getmtime(cfg_path))

    configs.sort(key=score)
    for c in configs: print(" -", os.path.dirname(c))
    LORA_SRC = os.path.dirname(configs[-1])
    print("\nDa chon Adapter:", LORA_SRC)
    print("Cac file trong thu muc Adapter:")
    for n in sorted(os.listdir(LORA_SRC)):
        p = os.path.join(LORA_SRC, n)
        if os.path.isfile(p): print(f"   {n}  ({os.path.getsize(p)/1e6:.1f} MB)")
    cfg = json.load(open(os.path.join(LORA_SRC, "adapter_config.json"), encoding="utf-8"))
    print("\nbase_model_name_or_path ghi trong adapter:", cfg.get("base_model_name_or_path"))
"""))

cells.append(md("## Bước 6: Cài thư viện (chỉ Transformers + PEFT chuẩn, không cài lại Unsloth)"))
cells.append(code(r"""
import subprocess, sys
if EXISTING_Q:
    print("Bo qua - da co GGUF nen san.")
else:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "peft", "accelerate", "safetensors", "sentencepiece"], check=True)
    print("Cai xong.")
"""))

cells.append(md(r"""
## Bước 7: Gộp LoRA vào model gốc → thư mục model HF hoàn chỉnh
Đây là bước quan trọng nhất của cứu hộ: `merge_and_unload()` cộng trọng số LoRA vào model gốc để thành **một model độc lập**. Nếu `base_model_name_or_path` trong adapter trỏ tới đường dẫn cũ trong phiên train (đã mất), ô này tự thay bằng model Qwen2.5-3B-Instruct chính thức trên HuggingFace — cùng kiến trúc/trọng số gốc với bản của Unsloth nên kết quả gộp không đổi.
"""))
cells.append(code(r"""
import os, json, shutil

def merged_ready(d):
    if not os.path.exists(os.path.join(d, "config.json")): return False
    total = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".safetensors"))
    return total > 5e9

if EXISTING_Q or EXISTING_F16:
    print("Bo qua - khong can gop.")
elif merged_ready(MERGED_DIR):
    print("Da co san model da gop o", MERGED_DIR, "- bo qua (xoa thu muc nay neu muon gop lai).")
else:
    import torch, gc
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Chep Adapter ve dia local cho nhanh va on dinh hon doc tu Drive
    LORA_LOCAL = "/content/lora_input"
    if os.path.exists(LORA_LOCAL): shutil.rmtree(LORA_LOCAL)
    shutil.copytree(LORA_SRC, LORA_LOCAL)

    acfg = json.load(open(os.path.join(LORA_LOCAL, "adapter_config.json"), encoding="utf-8"))
    base_id = acfg.get("base_model_name_or_path", "") or BASE_FALLBACK
    if ("Qwen2.5-3B" in base_id) or (base_id.startswith("/") and not os.path.exists(base_id)):
        base_id = BASE_FALLBACK
    print("Base model se nap:", base_id)

    try:
        base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map="auto")
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="auto")
    print("Da nap base. Dang gan LoRA Adapter...")
    model = PeftModel.from_pretrained(base, LORA_LOCAL)
    print("Dang gop (merge_and_unload) - vai phut...")
    model = model.merge_and_unload()
    model.save_pretrained(MERGED_DIR, safe_serialization=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(LORA_LOCAL)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_id)
    tokenizer.save_pretrained(MERGED_DIR)

    del model, base
    gc.collect(); torch.cuda.empty_cache()
    print("\nDa luu model da gop vao", MERGED_DIR)
    for n in sorted(os.listdir(MERGED_DIR)):
        p = os.path.join(MERGED_DIR, n)
        if os.path.isfile(p): print(f"   {n}  ({os.path.getsize(p)/1e9:.2f} GB)")
    assert merged_ready(MERGED_DIR), "Thu muc da gop chua du file - hay gui log loi o tren de duoc ho tro."
"""))

cells.append(md(r"""
## Bước 8: Convert model đã gộp → GGUF 16-bit bằng llama.cpp (thủ công)
Clone `llama.cpp` và chạy script chuyển đổi chính thức của nó. File 16-bit tạo ra (~6 GB) **đã là GGUF dùng được ngay**; bước nén tiếp theo chỉ để file nhỏ hơn. Vậy nên dù bước sau có trục trặc, bạn vẫn không trắng tay.
"""))
cells.append(code(r"""
import os, subprocess, sys

def gguf_ok(p, min_gb=0.5):
    if not os.path.exists(p) or os.path.getsize(p) < min_gb * 1e9: return False
    with open(p, "rb") as f: return f.read(4) == b"GGUF"

if EXISTING_Q:
    F16_GGUF = EXISTING_Q; print("Bo qua - da co GGUF nen san.")
elif EXISTING_F16:
    F16_GGUF = EXISTING_F16; print("Dung file 16-bit san co:", F16_GGUF)
else:
    if not os.path.exists("/content/llama.cpp"):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp", "/content/llama.cpp"], check=True)
    req = "/content/llama.cpp/requirements.txt"
    if os.path.exists(req):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", req])
    if gguf_ok(F16_GGUF, 4):
        print("Da co san file F16 tu lan chay truoc - bo qua convert.")
    else:
        cmd = [sys.executable, "/content/llama.cpp/convert_hf_to_gguf.py", MERGED_DIR,
               "--outfile", F16_GGUF, "--outtype", "f16"]
        print("Chạy:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    assert gguf_ok(F16_GGUF, 4), "Convert chua ra file GGUF 16-bit hop le - hay gui log o tren de duoc ho tro."
    print(f"\nOK: {F16_GGUF} ({os.path.getsize(F16_GGUF)/1e9:.2f} GB), dau file GGUF hop le.")
"""))

cells.append(md(r"""
## Bước 9: Nén xuống Q4_K_M (~2 GB)
Biên dịch riêng công cụ `llama-quantize` (CPU là đủ) rồi nén. Đây chính là khâu từng thất bại bên trong hàm tiện ích của Unsloth ở notebook chính — làm thủ công thì lỗi (nếu có) hiện rõ nguyên văn để xử lý, và bạn đã cầm chắc file 16-bit ở Bước 8 nên không còn rủi ro mất trắng.
"""))
cells.append(code(r"""
import os, subprocess, glob

if EXISTING_Q:
    FINAL_GGUF = EXISTING_Q; print("Dung luon GGUF nen san co:", FINAL_GGUF)
elif gguf_ok(FINAL_GGUF, 1):
    print("Da co san file da nen tu lan chay truoc:", FINAL_GGUF)
else:
    if not os.path.exists("/content/llama.cpp"):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp", "/content/llama.cpp"], check=True)
    subprocess.run(["cmake", "-S", "/content/llama.cpp", "-B", "/content/llama.cpp/build",
                    "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF"], check=True)
    built = subprocess.run(["cmake", "--build", "/content/llama.cpp/build", "--target", "llama-quantize", "-j", "2"])
    if built.returncode != 0:
        subprocess.run(["cmake", "--build", "/content/llama.cpp/build", "--target", "quantize", "-j", "2"], check=True)
    cand = [p for p in glob.glob("/content/llama.cpp/build/bin/*quantize*") + glob.glob("/content/llama.cpp/build/*quantize*") if os.path.isfile(p) and os.access(p, os.X_OK)]
    assert cand, "Khong tim thay file thuc thi quantize sau khi build - hay gui log build o tren."
    print("Cong cu quantize:", cand[0])
    subprocess.run([cand[0], F16_GGUF, FINAL_GGUF, QUANT], check=True)
    assert gguf_ok(FINAL_GGUF, 1), "Nen xong nhung file khong hop le - hay gui log o tren."
    print(f"\nOK: {FINAL_GGUF} ({os.path.getsize(FINAL_GGUF)/1e9:.2f} GB)")
"""))

cells.append(md(r"""
## Bước 10: Chép sản phẩm lên Google Drive + kiểm chứng tại chỗ 🎉
Sau khi ô này chạy xong, hãy tự mở Drive → `MyDrive/ai-gia-su-toan/gguf/` và nhìn thấy tận mắt file `.gguf` khoảng 2 GB cùng `Modelfile` — đó mới là bằng chứng hoàn tất (đừng tin chỉ dòng chữ in trên Colab). Sau đó tải về máy và làm theo `HUONG_DAN_SU_DUNG_GGUF.md` trong repo.

**Nếu notebook này vẫn dừng ở bước nào:** đừng train lại. Hãy gửi lại (1) toàn bộ dòng lỗi đỏ ở ô bị dừng, và (2) ảnh chụp/danh sách file trong thư mục LoRA trên Drive. Chỉ hai thứ đó là đủ để lần ra đúng nguyên nhân.
"""))
cells.append(code(r"""
import os, shutil

SRC = FINAL_GGUF
assert gguf_ok(SRC, 1), f"File nguon chua hop le: {SRC}"
final_name = "gia-su-toan-qwen2.5-3b-q4_k_m.gguf"
dst_dir = os.path.join(DRIVE_PROJECT, "gguf")
os.makedirs(dst_dir, exist_ok=True)
dst = os.path.join(dst_dir, final_name)

if os.path.abspath(SRC) != os.path.abspath(dst):
    print(f"Dang chep {os.path.getsize(SRC)/1e9:.2f} GB len Drive - vai phut, dung tat tab...")
    shutil.copy(SRC, dst)
assert os.path.exists(dst) and os.path.getsize(dst) > 1e9, "Chep len Drive chua thanh cong!"
print(f"\nTHANH CONG tren Drive: {dst}\nKich thuoc: {os.path.getsize(dst)/1e9:.2f} GB")

modelfile = f'FROM ./{final_name}\nPARAMETER temperature 0.6\nPARAMETER top_p 0.9\nPARAMETER num_ctx 4096\nPARAMETER stop "<|im_end|>"\n'
open(os.path.join(dst_dir, "Modelfile"), "w", encoding="utf-8").write(modelfile)
print("Da ghi kem Modelfile (them dong SYSTEM day du tu file Modelfile trong repo khi dung Ollama).")

if TAI_THANG_VE_MAY:
    from google.colab import files
    files.download(dst)
else:
    print("Mo Google Drive ->", dst_dir, "de tai file ve may.")
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
out_path = os.path.join(ROOT, "CuuHo_Xuat_GGUF_Tu_LoRA_Colab.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Đã tạo", out_path, "số ô:", len(cells))

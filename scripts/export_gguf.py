#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge LoRA + Xuat GGUF (q4_k_m / q8_0) tu adapter da train bang Unsloth.
Theo docs Unsloth: model.save_pretrained_gguf("folder", tokenizer, quantization_method="q4_k_m")

Vi du:
    python scripts/export_gguf.py --adapter_dir outputs/lora --out_dir outputs/gguf --quant q4_k_m
    python scripts/export_gguf.py --adapter_dir outputs/lora --out_dir outputs/gguf --quant q8_0

File ket qua: outputs/gguf/*.gguf  (dung voi Ollama, LM Studio, llama.cpp, Jan, GPT4All...)
"""
import argparse, glob, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", default="outputs/lora", help="Folder LoRA da train (chua adapter_config.json)")
    ap.add_argument("--base_model", default="unsloth/Qwen2.5-3B-Instruct")
    ap.add_argument("--out_dir", default="outputs/gguf")
    ap.add_argument("--quant", default="q4_k_m", choices=["q4_k_m", "q8_0", "f16", "q5_k_m", "q6_k"])
    ap.add_argument("--max_seq_length", type=int, default=2048)
    args = ap.parse_args()

    from unsloth import FastLanguageModel

    # Cach 1 (khuyen dung): tai thang adapter folder - Unsloth tu doc base tu adapter_config
    # Neu loi, doi model_name thanh base_model roi dung PeftModel (xem ghi chu cuoi file)
    load_target = args.adapter_dir if os.path.exists(os.path.join(args.adapter_dir, "adapter_config.json")) else args.base_model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_target,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=False,   # xuat GGUF nen tai day du de gop trong so dung
    )
    FastLanguageModel.for_inference(model)  # quan trong: tranh xuat kem artifact luc train

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained_gguf(args.out_dir, tokenizer, quantization_method=args.quant)

    files = glob.glob(os.path.join(args.out_dir, "*.gguf"))
    print("\n=== FILE GGUF DA TAO ===")
    for f in sorted(files):
        print(f" - {f}  ({os.path.getsize(f)/1e9:.2f} GB)")
    print("\nChạy thử với Ollama: xem HUONG_DAN_SU_DUNG_GGUF.md (dùng kèm file Modelfile)")

if __name__ == "__main__":
    main()

# --- GHI CHU DU PHONG (neu save_pretrained_gguf Loi) -----------------------------------
# Buoc 1 - luu model da merge 16-bit:
#     model.save_pretrained_merged("outputs/merged_16bit", tokenizer, save_method="merged_16bit")
# Buoc 2 - convert bang llama.cpp thu cong:
#     git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && pip install -r requirements.txt
#     python convert_hf_to_gguf.py ../outputs/merged_16bit --outfile model-f16.gguf --outtype f16
#     make -j   (hoac cmake --build build)
#     ./llama-quantize model-f16.gguf gia-su-toan-q4_k_m.gguf Q4_K_M

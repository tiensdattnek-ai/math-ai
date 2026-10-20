#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune QLoRA AI Gia Sư Toán trên Qwen2.5-3B-Instruct bang Unsloth.
Chay tren Google Colab / may co GPU NVIDIA (>=12GB, khuyen nghi 16GB VRAM cho cau hinh mac dinh).

Vi du (train thu, nhanh):
    python scripts/train_lora.py --epochs 1 --output_dir outputs/lora
Vi du (tron them dataset mo, train ban chinh):
    python scripts/train_lora.py --extra_data data/open.jsonl --epochs 2 --output_dir outputs/lora
"""
import argparse, json, os

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="unsloth/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--extra_data", default=None, help="vd data/open.jsonl tu load_open_datasets.py")
    ap.add_argument("--val_data", default="data/val.jsonl")
    ap.add_argument("--output_dir", default="outputs/lora")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=16)
    args = ap.parse_args()

    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,            # tu chon bf16 neu GPU ho tro
        load_in_4bit=True,     # QLoRA: vua T4 16GB
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=True,
    )

    rows = load_jsonl(args.data)
    if args.extra_data and os.path.exists(args.extra_data):
        extra = load_jsonl(args.extra_data)
        rows += extra
        print(f"Tron them {len(extra)} mau tu dataset mo")
    print(f"Tong so mau train: {len(rows)}")

    def format_row(row):
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        # Unsloth/SFT se tự thêm EOS trong nhieu cau hình; bo sung thu cong de chac model biet dung
        if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
            text += tokenizer.eos_token
        return {"text": text}

    train_ds = Dataset.from_list([format_row(r) for r in rows])

    # TRL thay doi API giua cac phien ban -> xay dung tuong thich ca SFTConfig moi va cu
    import inspect
    base_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
        packing=False,
    )
    try:
        try:
            sft_args = SFTConfig(**base_kwargs, dataset_text_field="text", max_length=args.max_seq_length)
        except TypeError:
            sft_args = SFTConfig(**base_kwargs, dataset_text_field="text", max_seq_length=args.max_seq_length)
    except TypeError:
        sft_args = SFTConfig(**base_kwargs)

    trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters)
    t_kwargs = {"model": model, "train_dataset": train_ds, "args": sft_args}
    t_kwargs["processing_class" if "processing_class" in trainer_params else "tokenizer"] = tokenizer
    trainer = SFTTrainer(**t_kwargs)
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Da luu LoRA adapter vao {args.output_dir}")
    print("Bước tiep: python scripts/export_gguf.py --adapter_dir", args.output_dir)

if __name__ == "__main__":
    main()

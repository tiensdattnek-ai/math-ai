#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nạp dataset Toán mở (đã dịch tiếng Việt) trên HuggingFace va chuyển về cùng
format messages nhu dataset SGK tu sinh (scripts/generate_dataset.py).

Nguồn (đã kiểm chứng tồn tại):
- 5CD-AI/Vietnamese-microsoft-orca-math-word-problems-200k-gg-translated (200K dòng, Toán lời văn)
- 5CD-AI/Vietnamese-meta-math-MetaMathQA-40K-gg-translated (40K dòng)
- Dự phòng: microsoft/orca-math-word-problems-200k (tiếng Anh - vẫn giúp kỹ năng giải Toán)

Script tự dò tên cột câu hỏi/đáp án vì các bản dịch đặt tên cột không thống nhất.

Cách chạy (trên Colab):
    python scripts/load_open_datasets.py --orca_n 4000 --meta_n 4000 --out data/open.jsonl
"""
import argparse, json, sys

SYSTEM_PROMPT = (
    "Bạn là AI gia sư Toán học Việt Nam cho học sinh từ lớp 1 đến lớp 12 (ôn thi tốt nghiệp THPT). "
    "Bạn luôn trả lời bằng tiếng Việt dễ hiểu, đúng chương trình SGK Việt Nam, kiểm tra lại kết quả trước khi kết luận. "
    "Có 2 chế độ: (1) GIẢI CHI TIẾT: phân tích đề, giải từng bước, kết luận đáp án, cho bài luyện tương tự. "
    "(2) GỢI Ý (Socratic): KHÔNG cho đáp án ngay, chỉ đưa gợi ý và đặt câu hỏi gợi mở để học sinh tự suy luận. "
    "Nếu học sinh yêu cầu 'gợi ý' thì dùng chế độ 2, nếu yêu cầu 'giải chi tiết' thì dùng chế độ 1, nếu không nói rõ thì dùng chế độ 1. "
    "Luôn thân thiện, động viên học sinh, trình bày ký hiệu Toán rõ ràng."
)

Q_CANDIDATES = ["question_vi", "vi_question", "question", "input", "prompt", "instruction", "query", "problem"]
A_CANDIDATES = ["answer_vi", "vi_answer", "answer", "output", "response", "solution", "completion"]

def pick_col(cols, candidates):
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    # tìm cột chứa từ khóa
    for cand in candidates:
        for c in cols:
            if cand.split("_")[0] in c.lower():
                return c
    return None

def fetch(ds_id, n, source_name, max_chars=3000):
    from datasets import load_dataset
    out = []
    try:
        ds = load_dataset(ds_id, split="train", streaming=True)
    except Exception as e:
        print(f"[Bo qua] Khong tai duoc {ds_id}: {e}", file=sys.stderr)
        return out
    first = True
    qcol = acol = None
    for row in ds:
        if first:
            cols = list(row.keys())
            print(f"[{source_name}] cac cot: {cols}")
            qcol, acol = pick_col(cols, Q_CANDIDATES), pick_col(cols, A_CANDIDATES)
            print(f"[{source_name}] dung cot hoi={qcol}, cot dap={acol}")
            first = False
            if not qcol or not acol:
                print(f"[Bo qua] {ds_id}: khong tim thay cot phu hop", file=sys.stderr)
                return out
        q, a = str(row.get(qcol, "")).strip(), str(row.get(acol, "")).strip()
        if not q or not a or len(a) > max_chars or len(q) > 1500:
            continue
        out.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Giải chi tiết bài Toán sau giúp em:\n{q}"},
                {"role": "assistant", "content": a},
            ],
            "grade": 0, "mode": "giai_chi_tiet", "topic": f"open_dataset:{source_name}",
        })
        if len(out) >= n:
            break
    print(f"[{source_name}] lay duoc {len(out)} mau")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orca_n", type=int, default=4000)
    ap.add_argument("--meta_n", type=int, default=4000)
    ap.add_argument("--out", default="data/open.jsonl")
    args = ap.parse_args()

    rows = []
    if args.orca_n > 0:
        rows += fetch("5CD-AI/Vietnamese-microsoft-orca-math-word-problems-200k-gg-translated", args.orca_n, "orca_vi")
    if args.meta_n > 0:
        rows += fetch("5CD-AI/Vietnamese-meta-math-MetaMathQA-40K-gg-translated", args.meta_n, "metamath_vi")
    if not rows:
        print("Khong lay duoc dataset tieng Viet, thu ban tieng Anh du phong...", file=sys.stderr)
        rows += fetch("microsoft/orca-math-word-problems-200k", 4000, "orca_en")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Da ghi {len(rows)} mau vao {args.out}")

if __name__ == "__main__":
    main()

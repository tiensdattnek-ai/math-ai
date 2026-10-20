#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Xuất GGUF hoàn toàn tự động trên GitHub Actions (CPU runner, khong can GPU):
  1. Tim/tai LoRA Adapter (file zip tai len chat -> draft release, hoac URL, hoac folder trong repo)
  2. Gop LoRA vao base model bang Transformers + PEFT chuan (merge_and_unload)
  3. Convert HF -> GGUF F16 bang llama.cpp chinh thuc
  4. Nen Q4_K_M bang llama-quantize, kiem tra header GGUF + kich thuoc sau moi buoc
  5. Don dep file trung gian sau tung buoc de khong het dia runner

Chay boi .github/workflows/export-gguf.yml. Cung co the chay thu cong tren may co
~16GB RAM + ~25GB dia trong.
"""
import argparse, glob, hashlib, json, os, re, shutil, subprocess, sys, zipfile
from pathlib import Path

WORK = Path("/tmp/export-work")
OUT_DIR = Path("outputs")
FINAL_NAME = "gia-su-toan-qwen2.5-3b-q4_k_m.gguf"


def log(*a):
    print("\n==>", *a, flush=True)


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def disk():
    u = shutil.disk_usage("/")
    print(f"[disk] free {u.free/1e9:.1f} GB / total {u.total/1e9:.1f} GB", flush=True)


def gguf_ok(p: Path, min_gb=0.5) -> bool:
    if not p.exists() or p.stat().st_size < min_gb * 1e9:
        return False
    with open(p, "rb") as f:
        return f.read(4) == b"GGUF"


def find_adapter(root: Path) -> Path:
    configs = [p for p in root.rglob("adapter_config.json")]
    assert configs, f"Khong thay adapter_config.json trong {root}"

    def score(cfg: Path):
        d = cfg.parent
        m = re.search(r"checkpoint-(\d+)", str(d))
        has_w = any((d / n).exists() for n in ("adapter_model.safetensors", "adapter_model.bin"))
        return (has_w, 2 if d.name == "lora" else 0, int(m.group(1)) if m else -1, cfg.stat().st_mtime)

    configs.sort(key=score)
    for c in configs:
        print(" - adapter candidate:", c.parent)
    chosen = configs[-1].parent
    print("Da chon Adapter:", chosen)
    return chosen


def extract_zips(root: Path):
    for z in list(root.rglob("*.zip")):
        log("Giai nen", z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(z.parent)


def prepare_adapter(args) -> Path:
    download = WORK / "adapter_download"
    download.mkdir(parents=True, exist_ok=True)

    # Uu tien 1: folder Adapter da co ngay trong repo
    repo_candidate = Path("lora_adapter")
    if repo_candidate.exists():
        log("Dung Adapter trong repo:", repo_candidate)
        return find_adapter(repo_candidate)

    # Uu tien 2: URL truc tiep (link file .zip HOAC link THU MUC Google Drive da chia se)
    if args.adapter_url:
        if "drive.google.com" in args.adapter_url and "/folders/" in args.adapter_url:
            # Link thu muc: tai ca thu muc, find_adapter() se tu loc dung adapter can dung
            run([sys.executable, "-m", "gdown", "--folder", args.adapter_url, "-O", download])
        elif "drive.google.com" in args.adapter_url:
            run([sys.executable, "-m", "gdown", "--fuzzy", args.adapter_url, "-O", download / "adapter.zip"])
        else:
            run(["curl", "-L", "--fail", args.adapter_url, "-o", download / "adapter.zip"])

    # Uu tien 3: Asset trong GitHub Release (ke ca ban draft chua publish)
    else:
        env = dict(os.environ)
        assert env.get("GH_TOKEN"), "Thieu GH_TOKEN de tai Release asset"
        log(f"Tai Adapter tu GitHub Release tag '{args.adapter_release_tag}' cua {args.repo}")
        run(["gh", "release", "download", args.adapter_release_tag, "--repo", args.repo,
             "--dir", download, "--clobber"], env=env)

    extract_zips(download)
    return find_adapter(download)


def merge(adapter_dir: Path, base_override: str) -> Path:
    import torch, gc
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    base_id = base_override or cfg.get("base_model_name_or_path") or "Qwen/Qwen2.5-3B-Instruct"
    if "Qwen2.5-3B" in str(base_id) or (str(base_id).startswith("/") and not os.path.exists(base_id)):
        base_id = "Qwen/Qwen2.5-3B-Instruct"
    log("Base model:", base_id, "| Adapter config r/alpha/use_rslora:",
        cfg.get("r"), cfg.get("lora_alpha"), cfg.get("use_rslora"))

    # Chep Adapter ve vung lam viec de tranh loi duong dan/phien ban
    local_adapter = WORK / "adapter"
    if local_adapter.exists():
        shutil.rmtree(local_adapter)
    shutil.copytree(adapter_dir, local_adapter)

    log("Nap base model tren CPU (bf16, low_cpu_mem_usage)")
    try:
        base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

    log("Gan Adapter va gop trong so (merge_and_unload)")
    model = PeftModel.from_pretrained(base, str(local_adapter))
    model = model.merge_and_unload()

    merged = WORK / "merged_hf"
    if merged.exists():
        shutil.rmtree(merged)
    model.save_pretrained(str(merged), safe_serialization=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(local_adapter))
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_id)
    tokenizer.save_pretrained(str(merged))

    total = sum(p.stat().st_size for p in merged.glob("*.safetensors"))
    print(f"Merged safetensors: {total/1e9:.2f} GB")
    assert (merged / "config.json").exists() and total > 5e9, "Model da gop chua hoan chinh"

    del model, base
    gc.collect()
    # Don cache model goc de giai phong hang GB dia cho cac buoc sau
    hf_home = os.environ.get("HF_HOME")
    if hf_home and os.path.exists(hf_home):
        log("Xoa cache base model sau khi gop:", hf_home)
        shutil.rmtree(hf_home, ignore_errors=True)
    disk()
    return merged


def convert_and_quantize(merged: Path, quant: str) -> Path:
    llama = WORK / "llama.cpp"
    if not llama.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp", llama])
    req = llama / "requirements.txt"
    if req.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)])

    f16 = WORK / "model-f16.gguf"
    log("Convert HF -> GGUF F16")
    run([sys.executable, llama / "convert_hf_to_gguf.py", merged,
         "--outfile", f16, "--outtype", "f16"])
    assert gguf_ok(f16, 4), "File F16 GGUF khong hop le"
    print(f"F16 OK: {f16.stat().st_size/1e9:.2f} GB")

    log("Xoa thu muc merged (da co F16) de giai phong dia")
    shutil.rmtree(merged, ignore_errors=True)
    disk()

    log("Bien dich llama-quantize")
    run(["cmake", "-S", llama, "-B", llama / "build", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF"])
    built = subprocess.run(["cmake", "--build", str(llama / "build"), "--target", "llama-quantize", "-j", "2"])
    if built.returncode != 0:
        run(["cmake", "--build", llama / "build", "--target", "quantize", "-j", "2"])
    cand = [p for p in glob.glob(str(llama / "build" / "bin" / "*quantize*")) +
            glob.glob(str(llama / "build" / "*quantize*"))
            if os.path.isfile(p) and os.access(p, os.X_OK)]
    assert cand, "Khong thay binary quantize sau khi build"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final = OUT_DIR / FINAL_NAME
    log(f"Nen {quant}: {f16} -> {final}")
    run([cand[0], f16, final, quant])
    assert gguf_ok(final, 1), "File GGUF da nen khong hop le"
    print(f"FINAL OK: {final} ({final.stat().st_size/1e9:.2f} GB)")

    f16.unlink(missing_ok=True)
    return final


def finalize(final: Path):
    sha = hashlib.sha256()
    with open(final, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    info = {"file": final.name, "bytes": final.stat().st_size, "sha256": sha.hexdigest()}
    (OUT_DIR / "export_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (OUT_DIR / "SHA256SUMS.txt").write_text(f"{sha.hexdigest()}  {final.name}\n", encoding="utf-8")

    mf_src = Path("Modelfile")
    if mf_src.exists():
        txt = mf_src.read_text(encoding="utf-8")
        lines = [f"FROM ./{final.name}" if l.startswith("FROM ") else l for l in txt.splitlines()]
        (OUT_DIR / "Modelfile").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))
    disk()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-url", default="")
    ap.add_argument("--adapter-release-tag", default="lora-input")
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    ap.add_argument("--base-model", default="")
    ap.add_argument("--quant", default="Q4_K_M")
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    disk()
    adapter = prepare_adapter(args)
    merged = merge(adapter, args.base_model)
    final = convert_and_quantize(merged, args.quant)
    finalize(final)
    log("HOAN TAT", final)


if __name__ == "__main__":
    main()

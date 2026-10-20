#!/usr/bin/env python3
"""Dung notebook Webchat_Colab_Cloudflare.ipynb.
Notebook nay chi can Runtime > Run all tren tai khoan Google Colab cua nguoi dung:
cai Ollama, tai GGUF tu GitHub Release, chay webchat HTTP thuan (khong SSE),
roi mo Cloudflare Quick Tunnel va in link cong khai tam thoi.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.strip("\n").splitlines(True)}


cells = []

cells.append(md(r"""
# 🎓 AI Gia Sư Toán — Webchat tự động trên Google Colab + Cloudflare Tunnel

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tiensdattnek-ai/math-ai/blob/main/Webchat_Colab_Cloudflare.ipynb)

### Cách dùng duy nhất cần nhớ
1. Chọn **Runtime → Change runtime type → T4 GPU**.
2. Chọn **Runtime → Run all**.
3. Chờ khoảng **5–12 phút** ở lần đầu. Ô gần cuối sẽ in to dòng `LINK WEBCHAT:` cùng **mã truy cập 6 số**.
4. Mở link, nhập mã 6 số đó rồi chat như bình thường. Ô cuối phải tiếp tục chạy để phiên Colab còn sống; muốn tắt AI thì bấm nút dừng ở ô cuối.

Notebook tự làm toàn bộ: cài Ollama chính thức, tải GGUF Q4_K_M từ GitHub Release của repo, kiểm SHA256, tạo model `gia-su-toan`, chạy một webchat HTTP thuần ở cổng 7860, rồi mở Cloudflare Quick Tunnel. Quick Tunnel không cần tài khoản Cloudflare, nhưng **ai có link + mã truy cập đều dùng được GPU của phiên này** và link chết ngay khi Colab runtime dừng. Đừng đăng link/mã lên nơi công cộng.
"""))

cells.append(md("## 1. Cấu hình — mặc định là dùng được"))
cells.append(code(r"""
import os, random, string

GGUF_NAME = "gia-su-toan-qwen2.5-3b-q4_k_m.gguf"
GGUF_URL = f"https://github.com/tiensdattnek-ai/math-ai/releases/download/gguf-v1/{GGUF_NAME}"
EXPECTED_SHA256 = "76b80da7b692e727bc5f386a96d6354df2f531ebb313966a425db3a339d4f648"
MODEL_DIR = "/content/ai-gia-su"
MODEL_PATH = os.path.join(MODEL_DIR, GGUF_NAME)
MODELFILE_URL = "https://raw.githubusercontent.com/tiensdattnek-ai/math-ai/main/Modelfile"
WEBCHAT_PORT = 7860

# Ma 6 so ngau nhien bao mat cho moi phien de nguoi la du co link cung khong tu y dung GPU cua bạn.
CHAT_ACCESS_CODE = f"{secrets.randbelow(1_000_000):06d}"
print("Thu muc lam viec:", MODEL_DIR)
print("Ma truy cap webchat phien nay:", CHAT_ACCESS_CODE)
"""))

cells.append(md("## 2. Kiểm tra GPU"))
cells.append(code(r"""
import shutil, subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout or "Khong thay nvidia-smi")
free_root = shutil.disk_usage("/content").free / 1e9
print(f"Dung luong trong /content: {free_root:.1f} GB")
assert free_root > 10, "Can toi thieu khoang 10GB trong cho GGUF + model Ollama"
"""))

cells.append(md(r"""
## 3. Cài Ollama + Cloudflared
Ollama được cài bằng script Linux chính thức; Cloudflared được tải dạng binary từ GitHub Releases của Cloudflare. Không dán mật khẩu hay token cá nhân nào vào notebook.
"""))
cells.append(code(r"""
!apt-get update -qq && apt-get install -y -qq pciutils zstd curl
!curl -fsSL https://ollama.com/install.sh | sh
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared
!ollama --version
!cloudflared --version
"""))

cells.append(md("## 4. Tải GGUF từ GitHub Release và kiểm chứng SHA256"))
cells.append(code(r"""
import hashlib, os, requests, subprocess
os.makedirs(MODEL_DIR, exist_ok=True)

if not (os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000_000):
    print("Dang tai GGUF (~1.93 GB)...")
    subprocess.run(["curl", "-fL", "--retry", "3", "-C", "-", "-o", MODEL_PATH + ".tmp", GGUF_URL], check=True)
    os.replace(MODEL_PATH + ".tmp", MODEL_PATH)

assert open(MODEL_PATH, "rb").read(4) == b"GGUF", "File tai ve khong phai GGUF"
h = hashlib.sha256()
with open(MODEL_PATH, "rb") as f:
    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
        h.update(chunk)
sha = h.hexdigest()
print(f"GGUF: {os.path.getsize(MODEL_PATH)/1e9:.3f} GB")
print("SHA256:", sha)
assert sha == EXPECTED_SHA256, "SHA256 khong khop ban GGUF tren Release"

# Modelfile nam cung thu muc voi GGUF vi dong FROM dung duong dan tuong doi
resp = requests.get(MODELFILE_URL, timeout=60)
resp.raise_for_status()
open(os.path.join(MODEL_DIR, "Modelfile"), "w", encoding="utf-8").write(resp.text)
print("Da lay Modelfile.")
"""))

cells.append(md("## 5. Khởi động Ollama, tạo model và làm nóng GPU"))
cells.append(code(r"""
import json, os, subprocess, time, urllib.request, urllib.error

env = os.environ.copy()
# Thu vien NVIDIA san co cua Colab can duoc uu tien de Ollama nhan dung GPU.
env["LD_LIBRARY_PATH"] = "/usr/lib64-nvidia:" + env.get("LD_LIBRARY_PATH", "")
OLLAMA_LOG = open(os.path.join(MODEL_DIR, "ollama.log"), "ab")

def ollama_alive():
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False

if not ollama_alive():
    ollama_proc = subprocess.Popen(["ollama", "serve"], env=env, stdout=OLLAMA_LOG, stderr=subprocess.STDOUT)
for _ in range(90):
    if ollama_alive():
        break
    time.sleep(2)
assert ollama_alive(), "Ollama khong khoi dong duoc - xem /content/ai-gia-su/ollama.log"

print("Dang tao model gia-su-toan tu GGUF...")
subprocess.run(["ollama", "create", "gia-su-toan", "-f", "Modelfile"], cwd=MODEL_DIR, env=env, check=True)

# Goi thu mot cau ngan de model duoc nap truoc khi mo webchat.
payload = {
    "model": "gia-su-toan",
    "messages": [{"role": "user", "content": "Giải chi tiết: 2 + 3 bằng mấy?"}],
    "stream": False,
    "options": {"num_predict": 80, "temperature": 0.3},
}
req = urllib.request.Request(
    "http://127.0.0.1:11434/api/chat",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=300) as r:
    warm = json.loads(r.read().decode("utf-8"))
print("Test model OK:", warm["message"]["content"][:300])
"""))

cells.append(md(r"""
## 6. Chạy webchat HTTP thuần ở cổng 7860
Webchat này cố ý **không dùng SSE/WebSocket** vì Cloudflare Quick Tunnel không hỗ trợ SSE. Mỗi câu hỏi được gửi bằng một HTTP POST thường và nhận trọn câu trả lời, nên đi qua tunnel ổn định hơn. API chat bị khóa bằng mã 6 số ở ô cấu hình.
"""))
cells.append(code(r"""
import hmac, json, threading, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SYSTEM_PROMPT = (
    "Bạn là AI gia sư Toán học Việt Nam cho học sinh từ lớp 1 đến lớp 12 (ôn thi tốt nghiệp THPT). "
    "Luôn trả lời bằng tiếng Việt dễ hiểu, đúng chương trình SGK Việt Nam, kiểm tra lại kết quả trước khi kết luận. "
    "Có 2 chế độ: (1) GIẢI CHI TIẾT: phân tích đề, giải từng bước, kết luận đáp án, cho bài luyện tương tự. "
    "(2) GỢI Ý (Socratic): không cho đáp án ngay, chỉ gợi ý và đặt câu hỏi gợi mở. "
    "Nếu học sinh nói gợi ý thì dùng chế độ 2; nếu nói giải chi tiết thì dùng chế độ 1; nếu không rõ thì dùng chế độ 1."
)

HTML = r'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Gia Sư Toán</title>
<style>
body{font-family:system-ui,Arial,sans-serif;background:#f6f7fb;margin:0}.wrap{max-width:900px;margin:0 auto;height:100vh;display:flex;flex-direction:column}
header{background:#123c69;color:#fff;padding:12px 16px}header b{font-size:18px}.code{padding:8px 16px;background:#e8eef7;display:flex;gap:8px;align-items:center}
input,textarea,button{font:inherit;border-radius:10px;border:1px solid #bbb;padding:10px}button{background:#123c69;color:#fff;border:0;cursor:pointer}
#msgs{flex:1;overflow:auto;padding:16px}.msg{max-width:86%;padding:10px 12px;border-radius:14px;margin:8px 0;white-space:pre-wrap;line-height:1.5}
.user{background:#d9eaff;margin-left:auto}.bot{background:#fff;margin-right:auto;box-shadow:0 1px 4px #0002}.bar{display:flex;gap:8px;padding:10px;background:#fff}
textarea{flex:1;resize:none;height:48px}.hint{font-size:13px;color:#456}
</style></head><body><div class="wrap">
<header><b>🎓 AI Gia Sư Toán</b><div>Lớp 1 → Tốt nghiệp THPT</div></header>
<div class="code"><span>Mã truy cập:</span><input id="code" type="password" inputmode="numeric" placeholder="6 số trong Colab"><span class="hint">Mẹo: bắt đầu bằng “Giải chi tiết:” hoặc “Gợi ý cho em:”</span></div>
<div id="msgs"></div>
<form class="bar" id="form"><textarea id="q" placeholder="Nhập bài Toán… (Enter để gửi, Shift+Enter xuống dòng)"></textarea><button id="send">Gửi</button></form>
</div><script>
const msgs=document.getElementById('msgs'), q=document.getElementById('q'), code=document.getElementById('code'), send=document.getElementById('send');
code.value=sessionStorage.getItem('chat_code')||''; let history=[];
function add(cls,text){const d=document.createElement('div');d.className='msg '+cls;d.textContent=text;msgs.appendChild(d);msgs.scrollTop=msgs.scrollHeight;return d;}
add('bot','Xin chào em! Hãy nhập bài Toán. Thầy/cô có thể giải chi tiết hoặc chỉ gợi ý nếu em yêu cầu.');
q.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();document.getElementById('form').requestSubmit();}});
document.getElementById('form').addEventListener('submit',async e=>{e.preventDefault(); const text=q.value.trim(); if(!text)return;
 sessionStorage.setItem('chat_code',code.value.trim()); add('user',text); history.push({role:'user',content:text}); q.value=''; send.disabled=true;
 const wait=add('bot','⏳ AI đang giải…');
 try{const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json','X-Access-Code':code.value.trim()},body:JSON.stringify({message:text,history:history.slice(0,-1)})});
 const d=await r.json().catch(()=>({error:'Phan hoi khong hop le'})); if(!r.ok) throw new Error(d.error||('HTTP '+r.status));
 wait.textContent=d.reply; history.push({role:'assistant',content:d.reply});}
 catch(err){wait.textContent='⚠️ '+err.message;} finally{send.disabled=false;q.focus();}});
</script></body></html>'''

def clean_text(x, limit=12000):
    return str(x or "")[:limit]

def call_ollama(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in (history or [])[-12:]:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant"):
            messages.append({"role": item["role"], "content": clean_text(item.get("content"))})
    messages.append({"role": "user", "content": clean_text(message)})
    data = {
        "model": "gia-su-toan", "messages": messages, "stream": False,
        "options": {"temperature": 0.6, "top_p": 0.9, "num_ctx": 4096, "num_predict": 1024},
    }
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))["message"]["content"]

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, fmt, *args):
        print("[webchat]", self.address_string(), fmt % args)
    def reply(self, status, body, content_type="application/json; charset=utf-8"):
        raw = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.reply(200, HTML, "text/html; charset=utf-8")
        elif self.path == "/health":
            self.reply(200, json.dumps({"ok": True}))
        else:
            self.reply(404, json.dumps({"error": "Not found"}))
    def do_POST(self):
        if self.path != "/api/chat":
            self.reply(404, json.dumps({"error": "Not found"})); return
        supplied = self.headers.get("X-Access-Code", "")
        if CHAT_ACCESS_CODE and not hmac.compare_digest(supplied, CHAT_ACCESS_CODE):
            self.reply(401, json.dumps({"error": "Sai mã truy cập. Hãy nhập mã 6 số trong output của Colab."})); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000: raise ValueError("Du lieu gui len khong hop le")
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            answer = call_ollama(data.get("message", ""), data.get("history", []))
            self.reply(200, json.dumps({"reply": answer}, ensure_ascii=False))
        except Exception as exc:
            self.reply(500, json.dumps({"error": f"Loi khi goi model: {exc}"}, ensure_ascii=False))

web_server = ThreadingHTTPServer(("0.0.0.0", WEBCHAT_PORT), Handler)
threading.Thread(target=web_server.serve_forever, daemon=True).start()
with urllib.request.urlopen(f"http://127.0.0.1:{WEBCHAT_PORT}/health", timeout=10) as r:
    print("Webchat local OK:", r.read().decode())
print(f"Mo noi bo trong Colab: http://localhost:{WEBCHAT_PORT}")
"""))

cells.append(md("## 7. Mở Cloudflare Quick Tunnel và lấy link"))
cells.append(code(r"""
import re, subprocess, threading, time

tunnel_logs = []
public_url = None
url_ready = threading.Event()

tunnel_proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", f"http://localhost:{WEBCHAT_PORT}", "--no-autoupdate"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

def read_tunnel_output():
    global public_url
    for line in tunnel_proc.stdout:
        tunnel_logs.append(line.rstrip())
        m = re.search(r"https://[-a-zA-Z0-9]+\.trycloudflare\.com", line)
        if m and not public_url:
            public_url = m.group(0)
            url_ready.set()

threading.Thread(target=read_tunnel_output, daemon=True).start()
if not url_ready.wait(timeout=120):
    print("\n".join(tunnel_logs[-80:]))
    raise RuntimeError("Cloudflare chua cap link sau 120 giay - hay chay lai o nay mot lan")

print("\n" + "=" * 72)
print("🌐 LINK WEBCHAT:", public_url)
print("🔑 MÃ TRUY CẬP:", CHAT_ACCESS_CODE)
print("=" * 72)
print("Luu ý: ai có du ca link va ma deu dung duoc phien Colab nay. Dung chia se cong khai neu khong muon nguoi la dung GPU.")
"""))

cells.append(md(r"""
## 8. GIỮ PHIÊN SỐNG — để ô này tiếp tục chạy
Khi ô này còn quay, webchat còn hoạt động. Đóng tab trình duyệt của webchat thì không sao, nhưng **ngắt Colab runtime / dừng ô này / hết hạn phiên Colab** thì link Cloudflare sẽ chết. Đây là link demo tạm thời, không phải website 24/7.
"""))
cells.append(code(r"""
import time
print("Webchat dang chay tai:", public_url)
print("Ma truy cap:", CHAT_ACCESS_CODE)
try:
    while True:
        if tunnel_proc.poll() is not None:
            raise RuntimeError("Cloudflared da dung - hay chay lai o mo tunnel")
        time.sleep(60)
except KeyboardInterrupt:
    print("Da dung theo yeu cau.")
"""))

cells.append(md(r"""
## Xử lý lỗi nhanh
- **Link báo 502 trong 1–2 phút đầu:** chờ ô mô hình/webchat xong hẳn rồi tải lại. Nếu vẫn 502, chạy lại ô 7 để lấy một tunnel mới.
- **Báo sai mã:** mã nằm ở output của ô 1 và ô 7; mã thay đổi mỗi lần Run all từ đầu.
- **Câu trả lời quá lâu:** model đang chạy trên T4 của chính phiên Colab này. Hãy kiểm tra Runtime đang là GPU chứ không phải CPU, và tránh mở quá nhiều tab chat cùng lúc.
- **Colab tự ngắt:** không sửa được bằng code trong notebook. Mở lại notebook và Run all lại; model không cần train lại lần nào vì GGUF được tải từ GitHub Release.
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    },
    "cells": cells,
}
out_path = os.path.join(ROOT, "Webchat_Colab_Cloudflare.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Đã tạo", out_path, "số ô:", len(cells))

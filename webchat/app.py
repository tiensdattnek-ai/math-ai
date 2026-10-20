#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Webchat Gradio goi Ollama local (model gia-su-toan da tao tu GGUF)."""
import os
import requests
import gradio as gr

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.environ.get("OLLAMA_MODEL", "gia-su-toan")

SYSTEM_PROMPT = (
    "Bạn là AI gia sư Toán học Việt Nam cho học sinh từ lớp 1 đến lớp 12 "
    "(ôn thi tốt nghiệp THPT). Luôn trả lời bằng tiếng Việt dễ hiểu, đúng chương trình "
    "SGK Việt Nam, kiểm tra lại kết quả trước khi kết luận. Có 2 chế độ: "
    "(1) GIẢI CHI TIẾT: phân tích đề, giải từng bước, kết luận đáp án, cho bài luyện tương tự. "
    "(2) GỢI Ý (Socratic): không cho đáp án ngay, chỉ gợi ý và đặt câu hỏi gợi mở. "
    "Nếu học sinh nói 'gợi ý' thì dùng chế độ 2; nếu nói 'giải chi tiết' thì dùng chế độ 1; "
    "nếu không rõ thì dùng chế độ 1. Luôn thân thiện và động viên học sinh."
)


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return str(content)


def respond(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history or []:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant"):
            text = _content_to_text(item.get("content", "")).strip()
            if text:
                messages.append({"role": item["role"], "content": text})
    messages.append({"role": "user", "content": _content_to_text(message)})

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.6, "top_p": 0.9, "num_ctx": 4096},
            },
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return (
            "⚠️ Chưa kết nối được Ollama ở máy chủ Codespace. Hãy đợi khoảng 1 phút rồi gửi lại; "
            "nếu vẫn lỗi, mở Terminal trong Codespace và chạy `bash webchat/start.sh`."
        )
    except Exception as exc:  # hien loi ngan gon de nguoi dung gui lai khi can ho tro
        return f"⚠️ Có lỗi khi gọi model: `{exc}`"


demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="🎓 AI Gia Sư Toán — Lớp 1 đến Tốt nghiệp THPT",
    description=(
        "Thử hỏi: **Giải chi tiết phương trình x² - 7x + 12 = 0** hoặc "
        "**Gợi ý cho em bài này, đừng cho đáp án ngay: Tính ∫₀² 3x² dx**. "
        "Model 3B chạy CPU nên câu trả lời có thể mất một chút thời gian."
    ),
    textbox=gr.Textbox(placeholder="Nhập bài Toán của em…", submit_btn="Gửi"),
    fill_height=True,
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

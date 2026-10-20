# Mở webchat AI Gia Sư Toán bằng GitHub Codespaces

Đây là cách hợp lệ để có trải nghiệm gần giống “bấm một cái, cloud tự cài rồi đưa link webchat”, **không dùng GitHub Actions làm VPS/RDP**. Codespaces là môi trường cloud tương tác của GitHub; cổng web mặc định ở chế độ riêng tư, chỉ tài khoản tạo Codespace mới mở được sau khi đăng nhập. Nếu muốn chia sẻ cho người khác, bạn tự đổi Port Visibility trong tab **PORTS** và tự chịu trách nhiệm về việc công khai đó.

## Cách mở

1. Bấm: https://codespaces.new/tiensdattnek-ai/math-ai?quickstart=1
2. Giữ nguyên lựa chọn máy được đề xuất rồi tạo Codespace.
3. Chờ lần đầu khoảng **5–15 phút**: máy tự cài Ollama, tải GGUF khoảng 1,93 GB, tạo model và mở Gradio ở cổng `7860`.
4. Nếu trình duyệt không tự mở webchat, vào tab **PORTS** trong Codespace và bấm địa chỉ của cổng `7860` (AI Gia Su Toan Webchat).

## Lưu ý chi phí và vòng đời

- Tài khoản GitHub cá nhân có hạn mức Codespaces miễn phí hằng tháng; dùng quá hạn mức có thể bị giới hạn hoặc tính phí nếu bạn đã đặt phương thức thanh toán/giới hạn chi tiêu. Hãy tự kiểm tra Billing của tài khoản.
- Codespace tự dừng sau một thời gian không tương tác (mặc định 30 phút) và có thời gian sống tối đa mỗi lần chạy. Khi mở lại, `webchat/start.sh` sẽ tự khởi động lại Ollama + webchat; model đã tải vẫn còn nếu bạn chưa xóa Codespace.
- Đây là môi trường thử nghiệm/tương tác cá nhân, không phải hạ tầng webchat công khai 24/7. Muốn link công khai ổn định lâu dài, hãy triển khai lên dịch vụ hosting đúng mục đích như Hugging Face Spaces hoặc VPS riêng.
- Nếu Actions Artifact đã hết hạn và repo chưa có Release `gguf-v1`, bước tải model khi tạo Codespace mới sẽ báo lỗi; hãy xuất lại GGUF bằng workflow, đăng Release, hoặc đặt sẵn file GGUF vào `.local-ai/` trong Codespace.

# Bản đồ chương trình Toán lớp 1–12 (Việt Nam) và mức độ bao phủ của dataset

> Quan trọng — đọc để hiểu đúng kỳ vọng: fine-tune 3B chủ yếu dạy model **phong cách gia sư + định dạng lời giải tiếng Việt + 2 chế độ**, còn **kiến thức Toán nền** đến từ quá trình pretrain của Qwen2.5. Dataset trong project này bao phủ có hệ thống các **dạng Tính toán cốt lõi tự sinh được đáp án đúng bằng Python** ở mọi lớp. Các mảng nặng về chứng minh (hình học phẳng/không gian chứng minh, bất đẳng thức, bài toán cực trị khó của đề TN THPT phân loại cao) không thể tự sinh bằng mẫu — chúng được bổ trợ bởi dataset mở (Orca/MetaMath tiếng Việt) và kiến thức sẵn có của base model. Muốn nâng chất lượng lên mức sản phẩm thật: tăng dữ liệu giáo viên soạn + đề thi thật (có bản quyền rõ ràng) lên 50k–100k mẫu, xem mục "Nâng cấp" trong README.

| Lớp | Chủ đề SGK trọng tâm | Họ bài trong bộ sinh dataset này |
|---|---|---|
| 1 | Cộng/trừ ≤100, so sánh số, lời văn đơn giản | `g_l1_cong_tru`, `g_l1_so_sanh` |
| 2 | Cộng/trừ ≤1000, bảng nhân/chia 2–9 | `g_l2_nhan_chia`, `g_l2_cong_tru_1000` |
| 3 | Nhân/chia số lớn, chu vi, diện tích | `g_l3_chuvi_dientich` (+ nhân chia lớp 2 mở rộng) |
| 4 | Phân số, trung bình cộng, tổng–hiệu | `g_l4_phan_so_cong`, `g_l4_trung_binh_cong_tong_hieu` |
| 5 | Số thập phân (qua bài %), tỉ số phần trăm, vận tốc, thể tích | `g_l5_phan_tram`, `g_l5_van_toc`, `g_l5_the_tich` |
| 6 | Số nguyên, ƯCLN/BCNN, phân số nhân/chia | `g_l6_so_nguyen_ucln`, `g_l6_phan_so_nhan_chia` |
| 7 | Tỉ lệ thức, đa thức, lũy thừa, góc | `g_l7_ti_le_thuc_da_thuc`, `g_l7_luy_thua_goc` |
| 8 | Hằng đẳng thức, phân tích đa thức, PT/BPT bậc nhất, lập phương trình | `g_l8_hang_dang_thuc_pt`, `g_l8_bat_phuong_trinh`, `g_lap_phuong_trinh` |
| 9 | Căn, hệ PT, PT bậc 2 + Viète, hàm bậc nhất, hệ thức lượng | `g_l9_can_he_bac2`, `g_l9_ham_so_luong_giac_vuong` |
| 10 | Vectơ, BPT bậc 2, lượng giác cơ bản, phương trình đường thẳng | `g_l10_vecto_bpt2`, `g_l10_duong_thang` |
| 11 | Cấp số cộng/nhân, tổ hợp, xác suất, giới hạn, đạo hàm | `g_l11_capso_tohop`, `g_l11_xac_suat_dao_ham`, `g_l11_gioi_han` |
| 12 | Đạo hàm & cực trị, mũ–logarit, tích phân, số phức, thể tích chóp, Oxyz | `g_l12_cuc_tri_dao_ham`, `g_l12_mu_log`, `g_l12_tich_phan_so_phuc`, `g_l12_oxyz_the_tich` |

## Hai chế độ gia sư
- **Giải chi tiết (65% dữ liệu):** Phân tích đề → giải từng bước → đáp án → bài luyện tương tự.
- **Gợi ý Socratic (35% dữ liệu):** Chỉ gợi ý + đặt câu hỏi, **không lộ đáp án**. Kích hoạt khi học sinh nói "gợi ý/gợi mở".

## Cách mở rộng dataset
1. Viết thêm hàm `g_...()` mới trong `scripts/generate_dataset.py` theo cùng khuôn mẫu (luôn Tính án bằng `math`/`Fraction`), đăng ký trong `REGISTRY`.
2. Chạy lại `python scripts/generate_dataset.py --n 20000` để sinh bản lớn hơn.
3. Có đề bài/lời giải của riêng bạn (JSONL/CSV): viết đoạn chuyển đổi về format `messages` như mẫu trong `scripts/load_open_datasets.py`.

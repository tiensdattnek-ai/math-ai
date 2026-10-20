#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sinh dataset AI Gia Sư Toán Việt Nam - Lớp 1 đến Lớp 12 (Tốt nghiệp THPT)
- 48 họ bài toán mẫu (templates), mỗi họ sinh số ngẫu nhiên -> hàng nghìn bài khác nhau
- Đáp án được PYTHON TÍNH (math/Fraction) -> chính xác 100%, model không học đáp án sai
- 2 chế độ: "giai_chi_tiet" (giải từng bước + đáp án) và "goi_y" (Socratic, KHÔNG lộ đáp án)
- Output: JSONL dạng {"messages":[...], "grade":..., "topic":..., "mode":...}
  dùng trực tiếp cho Unsloth / TRL SFTTrainer với chat template Qwen.

Cách chạy:
    python scripts/generate_dataset.py --n 6000 --out_dir data
"""
import argparse, json, math, random, os
from fractions import Fraction
GCD = math.gcd  # Ước chung lớn nhất - dùng alias cho gọn

SYSTEM_PROMPT = (
    "Bạn là AI gia sư Toán học Việt Nam cho học sinh từ lớp 1 đến lớp 12 (ôn thi tốt nghiệp THPT). "
    "Bạn luôn trả lời bằng tiếng Việt dễ hiểu, đúng chương trình SGK Việt Nam, kiểm tra lại kết quả trước khi kết luận. "
    "Có 2 chế độ: (1) GIẢI CHI TIẾT: phân tích đề, giải từng bước, kết luận đáp án, cho bài luyện tương tự. "
    "(2) GỢI Ý (Socratic): KHÔNG cho đáp án ngay, chỉ đưa gợi ý và đặt câu hỏi gợi mở để học sinh tự suy luận. "
    "Nếu học sinh yêu cầu 'gợi ý' thì dùng chế độ 2, nếu yêu cầu 'giải chi tiết' thì dùng chế độ 1, nếu không nói rõ thì dùng chế độ 1. "
    "Luôn thân thiện, động viên học sinh, trình bày ký hiệu Toán rõ ràng."
)

R = random.Random(42)

def fmt(x):
    """Format số: bỏ .0 thừa, phân số -> a/b"""
    if isinstance(x, Fraction):
        return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
    if isinstance(x, float):
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:.2f}".rstrip("0").rstrip(".")
    return str(x)

def ri(a, b): return R.randint(a, b)
def pick(lst): return R.choice(lst)

NAMES = ["Minh", "An", "Linh", "Hoa", "Tuấn", "Mai", "Khánh", "Ngọc", "Hùng", "Trang", "Bảo", "Thảo"]
ITEMS = [("viên kẹo", "kẹo"), ("quả táo", "táo"), ("quyển vở", "vở"), ("cái bút", "bút"), ("bông hoa", "hoa"), ("viên bi", "bi")]

# Mỗi generator trả về: dict(problem, analysis, steps[], answer, hints[], ask, practice)
# ============================================================ LỚP 1-2
def g_l1_cong_tru():
    a, b = ri(6, 85), ri(5, 60)
    if R.random() < 0.5:
        name, (unit, _) = pick(NAMES), pick(ITEMS)
        c = ri(3, 40)
        return dict(problem=f"{name} có {a} {unit}. {name} được tặng thêm {c} {unit} nữa. Hỏi {name} có tất cả bao nhiêu {unit}?",
            analysis="Được tặng thêm tức là số lượng tăng lên, ta dùng phép cộng.",
            steps=[f"Số {unit} lúc sau = số {unit} lúc đầu + số được tặng thêm", f"= {a} + {c} = {a+c}"],
            answer=f"{a+c} {unit}",
            hints=["Khi được 'tặng thêm', số lượng sẽ tăng lên hay giảm đi?", "Ta cần cộng số ban đầu với số được tặng thêm."],
            ask=f"Em thử tính {a} + {c} xem được bao nhiêu?",
            practice=f"Lớp 1A có {b} bạn, thêm {ri(2,9)} bạn chuyển đến. Hỏi lớp có bao nhiêu bạn?")
    total = a + b
    return dict(problem=f"Tính: {total} - {b} = ?",
        analysis="Đây là phép trừ hai số. Ta có thể nghĩ: số nào cộng với số trừ sẽ bằng số bị trừ?",
        steps=[f"Ta thấy {a} + {b} = {total}", f"Vậy {total} - {b} = {a}"],
        answer=f"{a}",
        hints=["Hãy nghĩ ngược lại bằng phép cộng: mấy cộng với số trừ thì bằng số bị trừ?", f"Thử kiểm tra bằng phép cộng nhé."],
        ask=f"Số nào cộng với {b} thì bằng {total} nhỉ?",
        practice=f"Tính: {total} - {a} = ?")

def g_l1_so_sanh():
    a, b = ri(11, 99), ri(11, 99)
    if a == b: b += 3
    sign = ">" if a > b else "<"
    bigger, smaller = max(a, b), min(a, b)
    return dict(problem=f"Điền dấu >, < hoặc = thích hợp: {a} ... {b}",
        analysis="So sánh hai số có hai chữ số: so sánh chữ số hàng chục trước, nếu bằng nhau mới so sánh hàng đơn vị.",
        steps=[f"Số {a} có {a//10} chục và {a%10} đơn vị", f"Số {b} có {b//10} chục và {b%10} đơn vị",
               f"So sánh thấy {bigger} lớn hơn {smaller}"],
        answer=f"{a} {sign} {b}",
        hints=["Hãy so sánh chữ số hàng chục của hai số trước nhé.", "Số nào có hàng chục lớn hơn thì số đó lớn hơn."],
        ask=f"Hàng chục của {a} là mấy, hàng chục của {b} là mấy?",
        practice=f"Điền dấu thích hợp: {ri(20,89)} ... {ri(20,89)}")

def g_l2_nhan_chia():
    a, b = ri(2, 9), ri(2, 9)
    if R.random() < 0.5:
        groups = ri(2, 9)
        return dict(problem=f"Có {groups} hộp, mỗi hộp có {a} cái bánh. Hỏi có tất cả bao nhiêu cái bánh?",
            analysis="Mỗi hộp có số bánh như nhau, ta dùng phép nhân: số hộp nhân số bánh mỗi hộp.",
            steps=[f"Số bánh tất cả = {groups} × {a}", f"Dùng bảng nhân {a}: {groups} × {a} = {groups*a}"],
            answer=f"{groups*a} cái bánh",
            hints=["Đây là cộng lặp lại nhiều lần số bánh của mỗi hộp.", "Phép tính nào thay thế được phép cộng lặp lại nhỉ?"],
            ask=f"Em hãy viết phép nhân tương ứng với {groups} hộp, mỗi hộp {a} cái bánh xem?",
            practice=f"Có {b} túi, mỗi túi {ri(3,8)} viên bi. Có tất cả bao nhiêu viên bi?")
    total = a * b
    return dict(problem=f"Chia đều {total} chiếc kẹo cho {a} bạn. Hỏi mỗi bạn được bao nhiêu chiếc kẹo?",
        analysis="Chia đều tức là mỗi bạn được số kẹo bằng nhau, ta dùng phép chia.",
        steps=[f"Số kẹo mỗi bạn = {total} : {a}", f"Vì {a} × {b} = {total} nên {total} : {a} = {b}"],
        answer=f"{b} chiếc kẹo",
        hints=["Hãy dựa vào bảng nhân để tìm kết quả của phép chia.", f"Mấy nhân với {a} thì bằng {total}?"],
        ask=f"Trong bảng nhân {a}, kết quả nào bằng {total}?",
        practice=f"Chia đều {ri(12,40)} quyển sách cho {ri(2,6)} bạn. Mỗi bạn được mấy quyển?")

def g_l2_cong_tru_1000():
    a, b = ri(120, 700), ri(50, 400)
    op = R.choice(["+", "-"])
    if op == "-" and b > a: a, b = b + ri(10, 200), b
    r = a + b if op == "+" else a - b
    return dict(problem=f"Tính: {a} {op} {b} = ?",
        analysis=f"Đặt tính theo cột dọc: hàng đơn vị thẳng hàng đơn vị, hàng chục thẳng hàng chục, hàng trăm thẳng hàng trăm, rồi tính từ phải sang trái.",
        steps=[f"Đặt tính {a} {op} {b} theo cột dọc", f"Tính lần lượt từ hàng đơn vị, hàng chục đến hàng trăm", f"Được kết quả {r}"],
        answer=f"{r}",
        hints=["Nhớ đặt các chữ số cùng hàng thẳng cột với nhau nhé.", "Tính từ phải sang trái, chú ý nhớ/mượn khi cần."],
        ask=f"Em thử đặt tính {a} {op} {b} và tính hàng đơn vị trước xem được mấy?",
        practice=f"Tính: {ri(150,800)} {op} {ri(40,300)} = ?")

# ============================================================ LỚP 3
def g_l3_chuvi_dientich():
    a, b = ri(3, 25), ri(2, min(20, 25))
    if R.random() < 0.5:
        return dict(problem=f"Một hình chữ nhật có chiều dài {a} cm, chiều rộng {b} cm. Tính chu vi và diện tích hình chữ nhật đó.",
            analysis="Chu vi hình chữ nhật = (dài + rộng) × 2. Diện tích = dài × rộng.",
            steps=[f"Chu vi = ({a} + {b}) × 2 = {a+b} × 2 = {2*(a+b)} (cm)", f"Diện tích = {a} × {b} = {a*b} (cm²)"],
            answer=f"Chu vi {2*(a+b)} cm, diện tích {a*b} cm²",
            hints=["Em còn nhớ công thức chu vi hình chữ nhật không?", "Diện tích hình chữ nhật tính bằng chiều dài nhân chiều rộng."],
            ask=f"Trước tiên em tính ({a} + {b}) × 2 được bao nhiêu?",
            practice=f"Hình chữ nhật dài {ri(5,20)} m, rộng {ri(3,12)} m. Tính chu vi và diện tích.")
    return dict(problem=f"Một hình vuông có cạnh {a} cm. Tính chu vi và diện tích hình vuông đó.",
        analysis="Chu vi hình vuông = cạnh × 4. Diện tích hình vuông = cạnh × cạnh.",
        steps=[f"Chu vi = {a} × 4 = {4*a} (cm)", f"Diện tích = {a} × {a} = {a*a} (cm²)"],
        answer=f"Chu vi {4*a} cm, diện tích {a*a} cm²",
        hints=["Hình vuông có 4 cạnh bằng nhau.", "Diện tích hình vuông bằng cạnh nhân với chính nó."],
        ask=f"Cạnh hình vuông là {a} cm, vậy 4 cạnh cộng lại dài bao nhiêu?",
        practice=f"Hình vuông cạnh {ri(4,18)} dm. Tính chu vi và diện tích.")

# ============================================================ LỚP 4
def g_l4_phan_so_cong():
    d1 = pick([2, 3, 4, 5, 6, 8, 10, 12])
    d2 = pick([x for x in [2,3,4,5,6,8,10,12] if x != d1])
    n1, n2 = ri(1, d1-1) if d1 > 2 else 1, ri(1, d2-1) if d2 > 2 else 1
    p1, p2 = Fraction(n1, d1), Fraction(n2, d2)
    s = p1 + p2
    lcm = abs(d1 * d2) // GCD(d1, d2)
    return dict(problem=f"Tính: {fmt(p1)} + {fmt(p2)}",
        analysis="Hai phân số khác mẫu, ta phải quy đồng mẫu số trước rồi mới cộng tử số, giữ nguyên mẫu số chung.",
        steps=[f"Mẫu số chung là {lcm}", f"Quy đồng: {fmt(p1)} = {n1*(lcm//d1)}/{lcm}, {fmt(p2)} = {n2*(lcm//d2)}/{lcm}",
               f"Cộng hai phân số cùng mẫu: ({n1*(lcm//d1)} + {n2*(lcm//d2)})/{lcm} = {n1*(lcm//d1)+n2*(lcm//d2)}/{lcm}",
               f"Rút gọn (nếu được): = {fmt(s)}"],
        answer=fmt(s),
        hints=["Hai phân số chưa cùng mẫu số, cần quy đồng trước.", f"Mẫu số chung có thể chọn là bội chung nhỏ nhất của {d1} và {d2}."],
        ask=f"Bội chung nhỏ nhất của {d1} và {d2} là bao nhiêu?",
        practice=f"Tính: {ri(1,3)}/{pick([4,5,6])} + {ri(1,2)}/{pick([3,4,8])}")

def g_l4_trung_binh_cong_tong_hieu():
    if R.random() < 0.5:
        nums = [ri(8, 60) for _ in range(3)]
        s, avg = sum(nums), Fraction(sum(nums), 3)
        return dict(problem=f"Tìm số trung bình cộng của các số: {nums[0]}, {nums[1]} và {nums[2]}.",
            analysis="Trung bình cộng = tổng các số chia cho số lượng các số.",
            steps=[f"Tổng ba số = {nums[0]} + {nums[1]} + {nums[2]} = {s}", f"Trung bình cộng = {s} : 3 = {fmt(avg)}"],
            answer=fmt(avg),
            hints=["Muốn tìm trung bình cộng, trước tiên phải tìm tổng.", "Sau đó lấy tổng chia cho số lượng số hạng."],
            ask="Tổng của ba số đã cho bằng bao nhiêu?",
            practice=f"Tìm trung bình cộng của {ri(10,50)}, {ri(10,50)}, {ri(10,50)}, {ri(10,50)}.")
    diff = pick([4, 6, 8, 10, 12, 14, 16, 20])
    small = ri(10, 60); big = small + diff; total = small + big
    return dict(problem=f"Tổng của hai số là {total}, hiệu của hai số là {diff}. Tìm hai số đó.",
        analysis="Bài toán tìm hai số khi biết tổng và hiệu: Số bé = (Tổng - Hiệu) : 2; Số lớn = Tổng - Số bé.",
        steps=[f"Số bé = ({total} - {diff}) : 2 = {total-diff} : 2 = {small}", f"Số lớn = {total} - {small} = {big}",
               f"Kiểm tra lại: {small} + {big} = {total} ✓, {big} - {small} = {diff} ✓"],
        answer=f"Số bé là {small}, số lớn là {big}",
        hints=["Đây là dạng toán tìm hai số khi biết tổng và hiệu.", "Số bé bằng (tổng trừ hiệu) rồi chia 2."],
        ask=f"Em thử tính ({total} - {diff}) : 2 xem ra số nào?",
        practice=f"Tổng hai số là {ri(50,120)}, hiệu là {pick([6,8,10,12])}. Tìm hai số.")

# ============================================================ LỚP 5
def g_l5_phan_tram():
    whole = pick([100, 200, 250, 400, 500, 600, 800, 1000, 1200, 2000])
    pct = pick([5, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75])
    part = whole * pct // 100
    ctx = pick([
        (f"Một trường học có {whole} học sinh, trong đó có {pct}% là học sinh nữ. Hỏi trường có bao nhiêu học sinh nữ?", "học sinh nữ"),
        (f"Một cửa hàng bán một chiếc áo giá {whole} nghìn đồng, giảm giá {pct}%. Hỏi số tiền được giảm là bao nhiêu nghìn đồng?", "nghìn đồng"),
        (f"Một mảnh đất có diện tích {whole} m², người ta dùng {pct}% diện tích để trồng rau. Hỏi diện tích trồng rau là bao nhiêu m²?", "m²"),
    ])
    return dict(problem=ctx[0],
        analysis=f"Muốn tìm {pct}% của {whole}, ta lấy {whole} × {pct} : 100 (hoặc {whole} × {fmt(Fraction(pct,100))}).",
        steps=[f"{pct}% của {whole} = {whole} × {pct} : 100", f"= {whole*pct} : 100 = {part}"],
        answer=f"{part} {ctx[1]}",
        hints=[f"{pct}% nghĩa là {pct} phần trong 100 phần.", "Lấy giá trị toàn thể nhân với tỉ số phần trăm rồi chia 100."],
        ask=f"1% của {whole} bằng bao nhiêu? Từ đó suy ra {pct}% bằng bao nhiêu?",
        practice=f"Tìm {pick([15,20,35,45])}% của {pick([300,600,900,1500])}.")

def g_l5_van_toc():
    kind = R.choice(["v", "s", "t"])
    v = pick([4, 5, 10, 12, 15, 20, 30, 40, 45, 50, 60]); t = pick([2, 3, 4, 5, 6,])
    s = v * t
    vehicle = pick(["Một ô tô", "Một xe máy", "Một người đi xe đạp", "Một đoàn tàu"])
    if kind == "v":
        return dict(problem=f"{vehicle} đi được quãng đường {s} km trong {t} giờ. Tính vận tốc của chuyển động đó.",
            analysis="Công thức: vận tốc = quãng đường : thời gian (v = s : t).",
            steps=[f"v = s : t = {s} : {t} = {v} (km/giờ)"],
            answer=f"{v} km/giờ",
            hints=["Ba đại lượng quãng đường, vận tốc, thời gian liên hệ: s = v × t.", "Muốn tìm v ta lấy s chia cho t."],
            ask=f"Em lấy {s} chia cho {t} được bao nhiêu?",
            practice=f"Xe đi {ri(80,240)} km trong {ri(2,5)} giờ. Tính vận tốc.")
    if kind == "s":
        return dict(problem=f"{vehicle} đi với vận tốc {v} km/giờ trong {t} giờ. Tính quãng đường đi được.",
            analysis="Công thức: quãng đường = vận tốc × thời gian (s = v × t).",
            steps=[f"s = v × t = {v} × {t} = {s} (km)"],
            answer=f"{s} km",
            hints=["Mỗi giờ đi được v km, đi trong t giờ thì được bao nhiêu km?", "Dùng công thức s = v × t."],
            ask=f"Em thử nhân {v} với {t} xem được bao nhiêu?",
            practice=f"Tàu chạy {ri(30,70)} km/giờ trong {ri(3,7)} giờ. Tính quãng đường.")
    return dict(problem=f"{vehicle} phải đi quãng đường {s} km với vận tốc {v} km/giờ. Hỏi cần bao nhiêu thời gian?",
        analysis="Công thức: thời gian = quãng đường : vận tốc (t = s : v).",
        steps=[f"t = s : v = {s} : {v} = {t} (giờ)"],
        answer=f"{t} giờ",
        hints=["Từ s = v × t, em suy ra công thức tính t như thế nào?", "Lấy quãng đường chia cho vận tốc."],
        ask=f"Em lấy {s} chia cho {v} được bao nhiêu?",
        practice=f"Đi quãng đường {ri(90,300)} km với vận tốc {pick([30,45,50,60])} km/giờ mất bao lâu?")

def g_l5_the_tich():
    a, b, c = ri(2, 15), ri(2, 12), ri(2, 10)
    return dict(problem=f"Một hình hộp chữ nhật có chiều dài {a} dm, chiều rộng {b} dm, chiều cao {c} dm. Tính thể tích của hình hộp đó.",
        analysis="Thể tích hình hộp chữ nhật = chiều dài × chiều rộng × chiều cao (V = a × b × c).",
        steps=[f"V = {a} × {b} × {c}", f"= {a*b} × {c} = {a*b*c} (dm³)"],
        answer=f"{a*b*c} dm³",
        hints=["Thể tích cho biết lượng không gian hình chiếm chỗ.", "Nhân cả ba kích thước với nhau."],
        ask=f"Em tính {a} × {b} trước được bao nhiêu, rồi nhân tiếp với {c}?",
        practice=f"Bể cá dài {ri(5,12)} dm, rộng {ri(3,8)} dm, cao {ri(3,7)} dm. Tính thể tích.")

# ============================================================ LỚP 6
def g_l6_so_nguyen_ucln():
    if R.random() < 0.5:
        a, b = ri(2, 40) * pick([-1, 1]), ri(2, 40) * pick([-1, 1])
        op = R.choice(["+", "-", "×"])
        r = a + b if op == "+" else a - b if op == "-" else a * b
        rule = {"+": "Cộng hai số nguyên: cùng dấu thì cộng giá trị tuyệt đối giữ nguyên dấu; khác dấu lấy giá trị tuyệt đối lớn trừ nhỏ, mang dấu của số lớn hơn.",
                "-": "Trừ số nguyên: a - b = a + (số đối của b).",
                "×": "Nhân hai số nguyên: cùng dấu cho kết quả dương, khác dấu cho kết quả âm."}[op]
        return dict(problem=f"Tính: ({a}) {op} ({b})",
            analysis=rule,
            steps=[f"({a}) {op} ({b}) = {r}"],
            answer=f"{r}",
            hints=[rule, "Xác định dấu của kết quả trước, rồi tính giá trị."],
            ask="Kết quả sẽ mang dấu dương hay âm? Vì sao?",
            practice=f"Tính: ({ri(-30,30)}) {op} ({ri(-25,25)})")
    m = ri(2, 9); x, y = m * ri(2, 12), m * ri(2, 12)
    if x == y: y += m
    ucln = GCD(x, y); bcnn = x * y // ucln
    what = R.choice(["ƯCLN", "BCNN"])
    ans = ucln if what == "ƯCLN" else bcnn
    return dict(problem=f"Tìm {what} của {x} và {y}.",
        analysis=("Phân tích mỗi số ra thừa số nguyên tố. ƯCLN = tích các thừa số nguyên tố chung với số mũ nhỏ nhất."
                  if what == "ƯCLN" else "Phân tích mỗi số ra thừa số nguyên tố. BCNN = tích các thừa số nguyên tố chung và riêng với số mũ lớn nhất."),
        steps=[f"Phân tích thừa số nguyên tố của {x} và {y}", f"ƯCLN({x},{y}) = {ucln}, BCNN({x},{y}) = {x}×{y}/ƯCLN = {bcnn}"],
        answer=f"{what}({x}, {y}) = {ans}",
        hints=["Bước 1 luôn là phân tích các số ra thừa số nguyên tố.", "ƯCLN lấy số mũ nhỏ nhất, BCNN lấy số mũ lớn nhất."],
        ask=f"Em phân tích {x} ra thừa số nguyên tố được gì?",
        practice=f"Tìm ƯCLN và BCNN của {ri(12,60)} và {ri(12,60)}.")

def g_l6_phan_so_nhan_chia():
    p1 = Fraction(ri(1, 9), ri(2, 9)); p2 = Fraction(ri(1, 9), ri(2, 9))
    if R.random() < 0.5:
        r = p1 * p2
        return dict(problem=f"Tính: {fmt(p1)} × {fmt(p2)}",
            analysis="Muốn nhân hai phân số: nhân tử với tử, mẫu với mẫu, rồi rút gọn.",
            steps=[f"{fmt(p1)} × {fmt(p2)} = ({p1.numerator}×{p2.numerator})/({p1.denominator}×{p2.denominator}) = {p1.numerator*p2.numerator}/{p1.denominator*p2.denominator}",
                   f"Rút gọn: = {fmt(r)}"],
            answer=fmt(r),
            hints=["Nhân phân số không cần quy đồng mẫu.", "Có thể rút gọn chéo trước khi nhân cho gọn."],
            ask=f"Tử nhân tử bằng mấy, mẫu nhân mẫu bằng mấy?",
            practice=f"Tính: {ri(1,7)}/{ri(2,8)} × {ri(1,7)}/{ri(2,8)}")
    r = p1 / p2
    return dict(problem=f"Tính: {fmt(p1)} : {fmt(p2)}",
        analysis="Chia một phân số cho một phân số = nhân với phân số nghịch đảo của số chia.",
        steps=[f"Phân số nghịch đảo của {fmt(p2)} là {p2.denominator}/{p2.numerator}",
               f"{fmt(p1)} : {fmt(p2)} = {fmt(p1)} × {p2.denominator}/{p2.numerator} = {fmt(r)}"],
        answer=fmt(r),
        hints=["'Chia là nhân nghịch đảo'.", "Đảo ngược tử và mẫu của số chia rồi đổi thành phép nhân."],
        ask=f"Phân số nghịch đảo của {fmt(p2)} là phân số nào?",
        practice=f"Tính: {ri(1,7)}/{ri(2,8)} : {ri(1,7)}/{ri(2,8)}")

# ============================================================ LỚP 7
def g_l7_ti_le_thuc_da_thuc():
    if R.random() < 0.5:
        a = ri(2, 12); x = ri(2, 20); b = ri(2, 12); d = b * x // GCD(b, 1)
        # dựng a/b = x/d sao cho d nguyên: chọn b | (b*x) luôn đúng tick d = b*x/a cần a|b*x -> chọn a là ước của b*x đơn giản: đặt a,b,x rồi d=b*x/a, lặp tới khi nguyên
        for _ in range(50):
            a, b, x = ri(2, 12), ri(2, 15), ri(2, 24)
            if (b * x) % a == 0:
                d = b * x // a; break
        else:
            a, b, x, d = 2, 4, 6, 12
        return dict(problem=f"Tìm x trong tỉ lệ thức: {a}/{b} = x/{d}",
            analysis="Từ tỉ lệ thức, tích hai ngoại tỉ bằng tích hai trung tỉ: a × d = b × x.",
            steps=[f"Ta có: {b} × x = {a} × {d} = {a*d}", f"x = {a*d} : {b} = {x}"],
            answer=f"x = {x}",
            hints=["Nhớ tính chất: tích chéo trong tỉ lệ thức bằng nhau.", f"Viết thành {b} × x = {a} × {d} rồi giải như tìm thừa số chưa biết."],
            ask=f"Tích chéo {a} × {d} bằng bao nhiêu?",
            practice=f"Tìm x: {ri(2,9)}/{ri(2,9)} = x/{ri(6,30)}")
    coefs = [ri(1, 6), ri(-8, 8) or 3, ri(-10, 10)]
    x0 = ri(-4, 5)
    val = coefs[0]*x0*x0 + coefs[1]*x0 + coefs[2]
    poly = f"{coefs[0]}x² {'+' if coefs[1]>=0 else '-'} {abs(coefs[1])}x {'+' if coefs[2]>=0 else '-'} {abs(coefs[2])}"
    return dict(problem=f"Tính giá trị của đa thức P(x) = {poly} tại x = {x0}.",
        analysis="Muốn tính giá trị đa thức tại một giá trị của biến, ta thay biến bằng số đó rồi thực hiện phép tính theo đúng thứ tự: luỹ thừa → nhân → cộng/trừ. Chú ý khi thay số âm phải đặt trong ngoặc.",
        steps=[f"Thay x = {x0} vào P(x)", f"P({x0}) = {coefs[0]}×({x0})² + ({coefs[1]})×({x0}) + ({coefs[2]})",
               f"= {coefs[0]*x0*x0} + ({coefs[1]*x0}) + ({coefs[2]}) = {val}"],
        answer=f"P({x0}) = {val}",
        hints=["Thay x bằng số đã cho ở mọi vị trí có x.", "Cẩn thận: bình phương của số âm cho kết quả dương."],
        ask=f"({x0})² bằng bao nhiêu nhỉ?",
        practice=f"Tính giá trị của {ri(2,5)}x² - {ri(2,7)}x + {ri(1,9)} tại x = {ri(-3,4)}.")

def g_l7_luy_thua_goc():
    if R.random() < 0.5:
        base = pick([2, 3, 5, 10]); m, n = ri(2, 7), ri(2, 6)
        kind = R.choice(["mul", "pow"])
        if kind == "mul":
            return dict(problem=f"Viết kết quả dưới dạng một luỹ thừa: {base}^{m} × {base}^{n}, rồi tính giá trị nếu cơ số nhỏ.",
                analysis="Khi nhân hai luỹ thừa cùng cơ số, ta giữ nguyên cơ số và cộng các số mũ: a^m × a^n = a^(m+n).",
                steps=[f"{base}^{m} × {base}^{n} = {base}^({m}+{n}) = {base}^{m+n}" + (f" = {base**(m+n)}" if base**(m+n) < 100000 else "")],
                answer=f"{base}^{m+n}" + (f" = {base**(m+n)}" if base**(m+n) < 100000 else ""),
                hints=["Hai luỹ thừa có cùng cơ số.", "Nhân luỹ thừa cùng cơ số thì cộng số mũ."],
                ask=f"Số mũ mới sẽ là {m} cộng với mấy?",
                practice=f"Viết gọn: {pick([2,3,7])}^{ri(2,6)} × {pick([2,3,7])}^{ri(2,5)} (chọn cùng cơ số)")
        val_exp = m * n
        return dict(problem=f"Viết kết quả dưới dạng một luỹ thừa: ({base}^{m})^{n}",
            analysis="Luỹ thừa của luỹ thừa: (a^m)^n = a^(m×n), giữ nguyên cơ số, nhân hai số mũ.",
            steps=[f"({base}^{m})^{n} = {base}^({m}×{n}) = {base}^{val_exp}"],
            answer=f"{base}^{val_exp}",
            hints=["Đây là luỹ thừa của một luỹ thừa.", "Giữ cơ số, nhân hai số mũ với nhau."],
            ask=f"Số mũ cuối cùng bằng {m} nhân với bao nhiêu?",
            practice=f"Viết gọn: ({pick([3,5,2])}^{ri(2,5)})^{ri(2,4)}")
    a = ri(25, 160)
    return dict(problem=f"Hai góc kề bù, một góc bằng {a}°. Tính góc còn lại. Góc còn lại là góc nhọn, góc vuông hay góc tù?",
        analysis="Hai góc kề bù có tổng số đo bằng 180°.",
        steps=[f"Góc còn lại = 180° - {a}° = {180-a}°",
               f"Vì {'0° < ' + str(180-a) + '° < 90° nên đó là góc nhọn' if 180-a < 90 else str(180-a)+'° > 90° nên đó là góc tù' if 180-a>90 else 'bằng 90° nên đó là góc vuông'}"],
        answer=f"{180-a}°" + (" (góc nhọn)" if 180-a < 90 else " (góc tù)" if 180-a > 90 else " (góc vuông)"),
        hints=["Hai góc kề bù nằm trên cùng một đường thẳng.", "Tổng của chúng bằng 180°."],
        ask=f"Tổng hai góc kề bù bằng bao nhiêu độ? Vậy góc còn lại = 180° - {a}° = ?",
        practice=f"Hai góc kề bù, một góc {ri(30,150)}°. Tính góc còn lại.")

# ============================================================ LỚP 8
def g_l8_hang_dang_thuc_pt():
    kind = R.choice(["square", "factor", "linear"])
    if kind == "square":
        a = ri(1, 9); b = ri(1, 9); sign = R.choice(["+", "-"])
        val = (a + b) ** 2 if sign == "+" else (a - b) ** 2
        return dict(problem=f"Khai triển hằng đẳng thức: (x {sign} {a})² và Tính ({b} {sign} {a})² bằng hằng đẳng thức.",
            analysis=f"Dùng hằng đẳng thức bình phương của một {'tổng: (a+b)² = a² + 2ab + b²' if sign=='+' else 'hiệu: (a-b)² = a² - 2ab + b²'}.",
            steps=[f"(x {sign} {a})² = x² {sign} {2*a}x + {a*a}",
                   f"({b} {sign} {a})² = {b*b} {sign} {2*b*a} + {a*a} = {val}"],
            answer=f"(x {sign} {a})² = x² {sign} {2*a}x + {a*a}; ({b} {sign} {a})² = {val}",
            hints=["Xác định đâu là số thứ nhất, đâu là số thứ hai trong hằng đẳng thức.", "Đừng quên số hạng giữa '2ab' nhé!"],
            ask="Trong (x + a)², số hạng ở giữa là gì?",
            practice=f"Khai triển: (2x + {ri(2,7)})²")
    if kind == "factor":
        a = ri(2, 12)
        return dict(problem=f"Phân tích đa thức thành nhân tử: x² - {a*a}",
            analysis="Nhận thấy đây là hiệu hai bình phương: a² - b² = (a - b)(a + b).",
            steps=[f"x² - {a*a} = x² - {a}²", f"= (x - {a})(x + {a})"],
            answer=f"(x - {a})(x + {a})",
            hints=[f"{a*a} bằng số nào bình phương?", "Dùng hằng đẳng thức hiệu hai bình phương."],
            ask=f"{a*a} = mấy² nhỉ?",
            practice=f"Phân tích thành nhân tử: x² - {pick([16,25,36,49,81,100])}")
    # phương trình bậc nhất ax + b = c, nghiệm nguyên
    x = ri(-8, 12); a = pick([2, 3, 4, 5, -2, -3]); b = ri(-15, 15); c = a * x + b
    return dict(problem=f"Giải phương trình: {a}x {'+' if b>=0 else '-'} {abs(b)} = {c}",
        analysis="Giải phương trình bậc nhất một ẩn: chuyển số hạng không chứa x sang vế phải (nhớ đổi dấu), rồi chia hai vế cho hệ số của x.",
        steps=[f"{a}x = {c} {'-' if b>=0 else '+'} {abs(b)}", f"{a}x = {c-b}", f"x = {c-b} : {a} = {x}",
               f"Thử lại: {a}×({x}) {'+' if b>=0 else '-'} {abs(b)} = {a*x+b} ✓"],
        answer=f"x = {x}",
        hints=["Mục tiêu là cô lập x ở một vế.", "Khi chuyển vế một số hạng phải đổi dấu."],
        ask=f"Chuyển {b} sang vế phải, vế trái còn lại gì?",
        practice=f"Giải phương trình: {pick([3,4,6])}x - {ri(3,12)} = {ri(5,30)}")

def g_l8_bat_phuong_trinh():
    x = ri(-5, 10); a = pick([2, 3, 4]); b = ri(-10, 10); c = a * x + b
    sign = R.choice([">", "<"])
    return dict(problem=f"Giải bất phương trình và biểu diễn tập nghiệm trên trục số: {a}x {'+' if b>=0 else '-'} {abs(b)} {sign} {c}",
        analysis="Giải bất phương trình bậc nhất tương tự phương trình. Lưu ý đặc biệt: khi nhân/chia hai vế cho số ÂM thì bất đẳng thức đổi chiều (ở bài này hệ số của x dương nên giữ nguyên chiều).",
        steps=[f"{a}x {sign} {c} {'-' if b>=0 else '+'} {abs(b)}", f"{a}x {sign} {c-b}",
               f"x {sign} {c-b} : {a} = {x}", f"Biểu diễn trên trục số: {'gạch bỏ phần bên trái điểm ' + str(x) + ', dùng ngoặc ) tại ' + str(x) if sign=='>' else 'gạch bỏ phần bên phải điểm ' + str(x) + ', dùng ngoặc ( tại ' + str(x)}"],
        answer=f"x {sign} {x}",
        hints=["Các bước giống giải phương trình bậc nhất.", "Chỉ khác nhau khi chia cho số âm mới phải đổi chiều bất đẳng thức."],
        ask=f"Sau khi chuyển vế, em được {a}x {sign} bao nhiêu?",
        practice=f"Giải bất phương trình: {pick([2,5])}x + {ri(2,9)} > {ri(10,25)}")

# ============================================================ LỚP 9
def g_l9_can_he_bac2():
    kind = R.choice(["sqrt", "system", "quad"])
    if kind == "sqrt":
        a = ri(2, 7); b = pick([2, 3, 5, 6, 7, 10])
        radicand = a * a * b
        return dict(problem=f"Rút gọn biểu thức: √{radicand}",
            analysis="Đưa thừa số chính phương ra ngoài dấu căn: √(a²×b) = a×√b với a > 0.",
            steps=[f"Phân tích: {radicand} = {a*a} × {b} = {a}² × {b}", f"√{radicand} = √({a}²×{b}) = {a}√{b}"],
            answer=f"{a}√{b}",
            hints=[f"Tìm trong {radicand} một thừa số là số chính phương (4, 9, 16, 25...).", "√(a²·b) = a·√b."],
            ask=f"{radicand} chia hết cho số chính phương lớn nhất nào?",
            practice=f"Rút gọn: √{pick([18,20,27,32,45,48,50,72,75])}")
    if kind == "system":
        x, y = ri(-5, 8), ri(-5, 8)
        a1, b1 = pick([1, 2, 3]), pick([1, 2, -1]); a2, b2 = pick([1, 2, -2]), pick([1, 3, -1])
        if a1 * b2 == a2 * b1: a2 += 1
        c1, c2 = a1 * x + b1 * y, a2 * x + b2 * y
        def lin(a, b, c): return f"{a}x {'+' if b>=0 else '-'} {abs(b)}y = {c}"
        return dict(problem=f"Giải hệ phương trình: {{ {lin(a1,b1,c1)} ; {lin(a2,b2,c2)} }}",
            analysis="Dùng phương pháp cộng đại số hoặc phương pháp thế: khử một ẩn, giải phương trình một ẩn còn lại, rồi thay ngược lại tìm ẩn kia. Cuối cùng thử lại nghiệm vào cả hai phương trình.",
            steps=[f"Áp dụng phương pháp cộng đại số/thế để khử một ẩn", f"Giải được x = {x}",
                   f"Thay x = {x} vào một phương trình ban đầu, tìm được y = {y}",
                   f"Thử lại: {a1}×({x}) + ({b1})×({y}) = {c1} ✓ ; {a2}×({x}) + ({b2})×({y}) = {c2} ✓"],
            answer=f"(x; y) = ({x}; {y})",
            hints=["Em muốn khử ẩn x hay ẩn y trước? Hãy nhân các phương trình để hệ số của ẩn đó bằng nhau hoặc đối nhau.", "Sau khi tìm được một ẩn, thay lại để tìm ẩn còn lại."],
            ask="Nếu cộng/trừ hai phương trình, ẩn nào sẽ bị triệt tiêu?",
            practice="Giải hệ: { 2x + y = 7 ; x - y = 2 }")
    r1, r2 = ri(-8, 8), ri(-8, 8)
    if r1 == r2: r2 += 2
    s, p = -(r1 + r2), r1 * r2
    return dict(problem=f"Giải phương trình bậc hai: x² {'+' if s>=0 else '-'} {abs(s)}x {'+' if p>=0 else '-'} {abs(p)} = 0",
        analysis="Phân tích vế trái thành nhân tử hoặc dùng hệ thức Viète: tìm hai số có tổng = S, tích = P thì đó là hai nghiệm. Cũng có thể Tính biệt thức Δ = b² - 4ac.",
        steps=[f"Tìm hai số có tổng bằng {r1+r2} và tích bằng {p}: đó là {r1} và {r2}",
               f"Phân tích: (x {'-' if r1>=0 else '+'} {abs(r1)})(x {'-' if r2>=0 else '+'} {abs(r2)}) = 0",
               f"x = {r1} hoặc x = {r2}",
               f"Kiểm tra Viète: tổng hai nghiệm = {r1+r2} = -b/a ✓, tích = {p} = c/a ✓"],
        answer=f"x = {r1} hoặc x = {r2}",
        hints=["Thử nhẩm nghiệm bằng Viète: tìm hai số nhân lại bằng hệ số tự do và cộng lại bằng -(hệ số của x).", "Nếu nhẩm không được, hãy Tính Δ = b² - 4ac."],
        ask=f"Hai số nào nhân lại bằng {p} và cộng lại bằng {r1+r2}?",
        practice=f"Giải phương trình: x² - {ri(3,9)}x + {ri(4,20)} = 0")

def g_l9_ham_so_luong_giac_vuong():
    if R.random() < 0.5:
        k = ri(1, 5); doi, ke, huyen = 3 * k, 4 * k, 5 * k
        return dict(problem=f"Tam giác ABC vuông tại A có AB = {ke} cm, AC = {doi} cm, BC = {huyen} cm. Tính sin B, cos B, tan B.",
            analysis="Trong tam giác vuông: sin = đối/huyền, cos = kề/huyền, tan = đối/kề (xét với góc B: cạnh đối là AC, cạnh kề là AB, cạnh huyền là BC).",
            steps=[f"sin B = AC/BC = {doi}/{huyen} = {fmt(Fraction(doi,huyen))}",
                   f"cos B = AB/BC = {ke}/{huyen} = {fmt(Fraction(ke,huyen))}",
                   f"tan B = AC/AB = {doi}/{ke} = {fmt(Fraction(doi,ke))}"],
            answer=f"sin B = {fmt(Fraction(doi,huyen))}, cos B = {fmt(Fraction(ke,huyen))}, tan B = {fmt(Fraction(doi,ke))}",
            hints=["Nhớ mẹo: 'sin đi học, cos không hư, tan đoàn kết' - sin=đối/huyền, cos=kề/huyền, tan=đối/kề.", "Phải xác định đúng cạnh đối và cạnh kề của góc B."],
            ask="Với góc B, cạnh nào là cạnh đối, cạnh nào là cạnh kề?",
            practice="Tam giác vuông có hai cạnh góc vuông 6 và 8, cạnh huyền 10. Tính tỉ số lượng giác của góc nhọn nhỏ hơn.")
    a = pick([2, 3, -2, -1]); b = ri(-6, 6); x0 = ri(-3, 5); y0 = a * x0 + b
    return dict(problem=f"Cho hàm số y = {a}x {'+' if b>=0 else '-'} {abs(b)}. a) Hàm số đồng biến hay nghịch biến? Vì sao? b) Tính giá trị của y khi x = {x0}.",
        analysis="Hàm số bậc nhất y = ax + b đồng biến khi a > 0, nghịch biến khi a < 0. Muốn Tính y ta thay giá trị của x vào công thức.",
        steps=[f"a) Hệ số a = {a} {' > 0 nên hàm số đồng biến trên ℝ' if a>0 else ' < 0 nên hàm số nghịch biến trên ℝ'}",
               f"b) Thay x = {x0}: y = {a}×({x0}) + ({b}) = {y0}"],
        answer=f"{'Đồng biến' if a>0 else 'Nghịch biến'}; khi x = {x0} thì y = {y0}",
        hints=["Dấu của hệ số a quyết định tính đồng biến/nghịch biến.", "Thay trực tiếp giá trị x vào để Tính y."],
        ask=f"Hệ số a bằng bao nhiêu, âm hay dương?",
        practice=f"Cho y = -3x + {ri(1,8)}. Hàm đồng biến hay nghịch biến? Tính y khi x = 2.")

# ============================================================ LỚP 10
def g_l10_vecto_bpt2():
    kind = R.choice(["vec", "ineq2", "trig"])
    if kind == "vec":
        x1, y1, x2, y2 = ri(-6, 6), ri(-6, 6), ri(-6, 6), ri(-6, 6)
        sx, sy = x1 + x2, y1 + y2
        length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
        # khoảng cách giữa 2 điểm dùng cùng x1,y1,x2,y2 như tọa độ điểm cho sinh động
        return dict(problem=f"Cho hai vectơ a = ({x1}; {y1}) và b = ({x2}; {y2}). Tính tọa độ vectơ a + b và độ dài của vectơ a.",
            analysis="Cộng hai vectơ: cộng từng tọa độ tương ứng. Độ dài vectơ (x;y) là √(x²+y²).",
            steps=[f"a + b = ({x1}+({x2}); {y1}+({y2})) = ({sx}; {sy})",
                   f"|a| = √(({x1})² + ({y1})²) = √{x1*x1+y1*y1}" + (f" = {fmt(math.sqrt(x1*x1+y1*y1))}" if abs(math.sqrt(x1*x1+y1*y1)-round(math.sqrt(x1*x1+y1*y1)))<1e-9 else "")],
            answer=f"a + b = ({sx}; {sy}), |a| = √{x1*x1+y1*y1}",
            hints=["Cộng vectơ là cộng hoành với hoành, tung với tung.", "Độ dài vectơ dùng định lý Pythagoras."],
            ask="Hoành độ của a + b bằng gì?",
            practice=f"Cho u = ({ri(-4,4)}; {ri(-4,4)}), v = ({ri(-4,4)}; {ri(-4,4)}). Tính u + v và |u|.")
    if kind == "ineq2":
        r1, r2 = ri(-5, 3), ri(1, 7)
        if r1 >= r2: r1, r2 = r2 - 3, r2
        sign = R.choice(["> 0", "< 0"])
        sol = f"x < {r1} hoặc x > {r2}" if sign == "> 0" else f"{r1} < x < {r2}"
        s, p = -(r1 + r2), r1 * r2
        return dict(problem=f"Giải bất phương trình bậc hai: x² {'+' if s>=0 else '-'} {abs(s)}x {'+' if p>=0 else '-'} {abs(p)} {sign}",
            analysis="Tìm nghiệm của tam thức (cho bằng 0), rồi xét dấu: tam thức bậc hai hệ số a>0 mang dấu dương ngoài khoảng hai nghiệm, âm trong khoảng hai nghiệm ('trong trái dấu a, ngoài cùng dấu a').",
            steps=[f"x² {'+' if s>=0 else '-'} {abs(s)}x {'+' if p>=0 else '-'} {abs(p)} = 0 ⇔ x = {r1} hoặc x = {r2}",
                   f"Hệ số a = 1 > 0. Tam thức {'dương khi x ở ngoài khoảng ('+str(r1)+'; '+str(r2)+')' }",
                   f"Kết hợp điều kiện {sign}: tập nghiệm là {sol}"],
            answer=sol,
            hints=["Bước 1: giải phương trình cho tam thức bằng 0.", "Nhớ quy tắc xét dấu: 'trong trái - ngoài cùng' dấu với a."],
            ask="Tam thức có hai nghiệm là mấy?",
            practice=f"Giải bất phương trình: x² - {ri(2,7)}x + {ri(3,12)} < 0")
    # trig sin^2+cos^2 với bộ ba Pythagoras
    num, den = pick([(3, 5), (4, 5), (5, 13), (12, 13), (8, 17), (15, 17)])
    cos_num = int(math.sqrt(den * den - num * num))
    return dict(problem=f"Cho góc nhọn α với sin α = {num}/{den}. Tính cos α và tan α.",
        analysis="Dùng hệ thức lượng giác cơ bản sin²α + cos²α = 1. Vì α nhọn nên cos α > 0. Sau đó tan α = sin α / cos α.",
        steps=[f"cos²α = 1 - sin²α = 1 - ({num}/{den})² = 1 - {num*num}/{den*den} = {den*den-num*num}/{den*den} = {cos_num*cos_num}/{den*den}",
               f"cos α = {cos_num}/{den} (lấy giá trị dương vì α nhọn)",
               f"tan α = sin α / cos α = {num}/{den} : {cos_num}/{den} = {fmt(Fraction(num, cos_num))}"],
        answer=f"cos α = {cos_num}/{den}, tan α = {fmt(Fraction(num, cos_num))}",
        hints=["Hệ thức nào liên hệ sin và cos của cùng một góc?", "Chú ý điều kiện góc nhọn để chọn dấu cho cos."],
        ask="sin²α bằng bao nhiêu? Vậy cos²α = 1 - sin²α = ?",
        practice="Cho sin α = 5/13, α nhọn. Tính cos α, tan α.")

def g_l10_duong_thang():
    x1, y1 = ri(-4, 4), ri(-4, 4)
    m = pick([1, 2, 3, -1, -2]); dx = ri(1, 4); x2, y2 = x1 + dx, y1 + m * dx
    b = y1 - m * x1
    return dict(problem=f"Viết phương trình đường thẳng đi qua hai điểm A({x1}; {y1}) và B({x2}; {y2}).",
        analysis="Tìm vectơ chỉ phương AB rồi suy ra hệ số góc k, hoặc giả sử phương trình y = ax + b rồi thay tọa độ 2 điểm để giải hệ tìm a, b.",
        steps=[f"Vectơ chỉ phương AB = ({x2-x1}; {y2-y1}), hệ số góc k = ({y2}-{y1})/({x2}-{x1}) = {m}",
               f"Phương trình có dạng y - {y1} = {m}(x - {x1})",
               f"Rút gọn: y = {m}x {'+' if b>=0 else '-'} {abs(b)}",
               f"Kiểm tra điểm B: {m}×{x2} + ({b}) = {m*x2+b} = {y2} ✓"],
        answer=f"y = {m}x {'+' if b>=0 else '-'} {abs(b)}",
        hints=["Trước hết tìm hệ số góc bằng tỉ số hiệu tung độ và hiệu hoành độ.", "Dùng dạng điểm - hệ số góc: y - y₁ = k(x - x₁)."],
        ask="Hệ số góc của đường thẳng AB bằng bao nhiêu?",
        practice=f"Viết pt đường thẳng qua ({ri(-3,3)}; {ri(-3,3)}) và ({ri(0,5)}; {ri(-3,4)}).")

# ============================================================ LỚP 11
def g_l11_capso_tohop():
    kind = R.choice(["csc", "csn", "comb"])
    if kind == "csc":
        u1, d, n = ri(-5, 15), pick([2, 3, 4, 5, -2, -3]), ri(6, 20)
        un = u1 + (n - 1) * d; sn = n * (u1 + un) // 2 if (n * (u1 + un)) % 2 == 0 else Fraction(n * (u1 + un), 2)
        return dict(problem=f"Cho cấp số cộng có số hạng đầu u₁ = {u1}, công sai d = {d}. Tính số hạng thứ {n} (u{n}) và tổng {n} số hạng đầu tiên (S{n}).",
            analysis="Số hạng tổng quát: uₙ = u₁ + (n-1)d. Tổng n số hạng đầu: Sₙ = n(u₁+uₙ)/2.",
            steps=[f"u{n} = {u1} + ({n}-1)×({d}) = {u1} + {(n-1)*d} = {un}",
                   f"S{n} = {n}×({u1} + {un})/2 = {fmt(sn if isinstance(sn, Fraction) else Fraction(sn))}"],
            answer=f"u{n} = {un}, S{n} = {sn}",
            hints=["Áp dụng đúng công thức số hạng tổng quát của cấp số cộng.", "Lưu ý n-1 chứ không phải n khi nhân với công sai."],
            ask=f"u{n} = u₁ + ({n}-1)d, thay số vào được bao nhiêu?",
            practice=f"Cấp số cộng u₁ = {ri(1,10)}, d = {ri(2,5)}. Tính u₁₀ và S₁₀.")
    if kind == "csn":
        u1, q, n = pick([1, 2, 3, 5]), pick([2, 3, -2]), ri(4, 8)
        un = u1 * q ** (n - 1)
        return dict(problem=f"Cho cấp số nhân có số hạng đầu u₁ = {u1}, công bội q = {q}. Tính số hạng thứ {n} (u{n}).",
            analysis="Số hạng tổng quát của cấp số nhân: uₙ = u₁ × q^(n-1).",
            steps=[f"u{n} = {u1} × ({q})^({n}-1) = {u1} × {q**(n-1)} = {un}"],
            answer=f"u{n} = {un}",
            hints=["Phân biệt với cấp số cộng: ở đây mỗi số hạng nhân thêm công bội.", "Số mũ là n-1."],
            ask=f"({q})^{n-1} bằng bao nhiêu?",
            practice=f"Cấp số nhân u₁ = {pick([2,3])}, q = {pick([2,3])}. Tính u₆.")
    n = ri(6, 12); k = ri(2, 4)
    c = math.comb(n, k)
    return dict(problem=f"Một lớp có {n} học sinh. Hỏi có bao nhiêu cách chọn ra {k} học sinh để trực nhật (không phân biệt chức vụ)?",
        analysis="Chọn không phân biệt thứ tự/chức vụ -> dùng TỔ HỢP: C(n,k) = n! / (k!(n-k)!). (Nếu có phân biệt chức vụ mới dùng chỉnh hợp A(n,k)).",
        steps=[f"Số cách chọn = C({n},{k}) = {n}! / ({k}!×{n-k}!)", f"= {c} (cách)"],
        answer=f"{c} cách",
        hints=["Bài chọn người không xếp thứ tự thì dùng tổ hợp hay chỉnh hợp?", "C(n,k) = n!/(k!(n-k)!) - em có thể bấm máy bằng phím nCr."],
        ask="Việc chọn ở đây có quan tâm thứ tự không?",
        practice=f"Có {ri(8,15)} bạn, chọn {ri(2,3)} bạn làm tổ trưởng, tổ phó (có phân công). Có bao nhiêu cách?")

def g_l11_xac_suat_dao_ham():
    if R.random() < 0.5:
        red, blue = ri(3, 8), ri(3, 9); total = red + blue
        p = Fraction(red, total)
        return dict(problem=f"Một hộp chứa {red} viên bi đỏ và {blue} viên bi xanh (các viên bi giống hệt nhau về kích thước). Lấy ngẫu nhiên 1 viên bi. Tính xác suất để lấy được viên bi đỏ.",
            analysis="Xác suất cổ điển: P(A) = số kết quả thuận lợi cho A / tổng số kết quả đồng khả năng của phép thử.",
            steps=[f"Tổng số bi = {red} + {blue} = {total} -> số kết quả có thể = {total}",
                   f"Số kết quả thuận lợi (được bi đỏ) = {red}",
                   f"P = {red}/{total} = {fmt(p)}" + (f" ≈ {float(p):.3f}" if p.denominator != 1 else "")],
            answer=f"P = {fmt(p)}",
            hints=["Trước tiên đếm tổng số viên bi trong hộp.", "Xác suất = số bi đỏ chia cho tổng số bi."],
            ask="Hộp có tất cả bao nhiêu viên bi?",
            practice=f"Gieo một con xúc xắc cân đối. Tính xác suất xuất hiện mặt có số chấm lẻ.")
    a, b, c = ri(1, 5), ri(-9, 9), ri(-8, 8); x0 = ri(-3, 4)
    deriv_at = 3 * a * x0 * x0 + 2 * b * x0  # f = a x^3 + b x^2 + c -> f' = 3a x^2 + 2b x
    return dict(problem=f"Cho hàm số f(x) = {a}x³ {'+' if b>=0 else '-'} {abs(b)}x² {'+' if c>=0 else '-'} {abs(c)}. Tính f'({x0}).",
        analysis="Dùng quy tắc đạo hàm của luỹ thừa: (xⁿ)' = n·xⁿ⁻¹, đạo hàm của hằng số bằng 0. Tính đạo hàm chung trước rồi thay x.",
        steps=[f"f'(x) = {3*a}x² {'+' if 2*b>=0 else '-'} {abs(2*b)}x",
               f"Thay x = {x0}: f'({x0}) = {3*a}×({x0})² + ({2*b})×({x0}) = {3*a*x0*x0} + ({2*b*x0}) = {deriv_at}"],
        answer=f"f'({x0}) = {deriv_at}",
        hints=["Đạo hàm của x³ là 3x², của x² là 2x.", "Tính f'(x) dạng tổng quát trước, thay số sau."],
        ask="f'(x) của hàm đã cho bằng biểu thức nào?",
        practice=f"Cho f(x) = {ri(2,4)}x³ - {ri(2,6)}x² + {ri(1,5)}x. Tính f'(1).")

def g_l11_gioi_han():
    a = ri(2, 7)
    return dict(problem=f"Tính giới hạn: lim (x→{a}) của (x² - {a*a})/(x - {a}).",
        analysis="Thay trực tiếp x = a cho dạng 0/0 (dạng vô định) nên phải biến đổi: phân tích tử thành nhân tử để rút gọn nhân tử (x-a) gây ra 0/0.",
        steps=[f"x² - {a*a} = (x - {a})(x + {a}) (hiệu hai bình phương)",
               f"(x²-{a*a})/(x-{a}) = x + {a} với x ≠ {a}",
               f"lim (x→{a}) (x + {a}) = {a} + {a} = {2*a}"],
        answer=f"{2*a}",
        hints=["Thay số trực tiếp được gì? Đó có phải dạng vô định không?", "Hãy phân tích tử số bằng hằng đẳng thức hiệu hai bình phương."],
        ask=f"x² - {a*a} phân tích thành tích hai nhân tử nào?",
        practice=f"Tính lim (x→3) của (x² - 9)/(x - 3).")

# ============================================================ LỚP 12
def g_l12_mu_log():
    if R.random() < 0.5:
        base = pick([2, 3, 5]); k = ri(3, 6); val = base ** k
        b0 = ri(1, 6); x = k + b0
        return dict(problem=f"Giải phương trình mũ: {base}^(x - {b0}) = {val}",
            analysis="Đưa hai vế về cùng cơ số. Với 0 < a ≠ 1: a^f(x) = a^g(x) ⇔ f(x) = g(x).",
            steps=[f"{val} = {base}^{k}, phương trình trở thành {base}^(x-{b0}) = {base}^{k}",
                   f"x - {b0} = {k}", f"x = {x}"],
            answer=f"x = {x}",
            hints=[f"{val} là luỹ thừa bậc mấy của {base}?", "Cùng cơ số thì hai số mũ bằng nhau."],
            ask=f"Viết {val} dưới dạng luỹ thừa cơ số {base} được {base} mũ mấy?",
            practice=f"Giải phương trình: {pick([3,4,9])}^x = {pick([27,64,81,125])}")
    base = pick([2, 3, 10]); k = ri(2, 6); val = base ** k if base != 10 else 10 ** k
    return dict(problem=f"Tính giá trị biểu thức: log_{base}({val}) và giải thích bằng định nghĩa logarit.",
        analysis="Theo định nghĩa: log_a(b) = c ⇔ a^c = b. Ta tìm số mũ cần nâng cơ số lên để được số trong logarit.",
        steps=[f"Ta thấy {base}^{k} = {val}", f"Do đó log_{base}({val}) = {k}"],
        answer=f"{k}",
        hints=[f"log cơ số {base} hỏi: {base} mũ mấy thì bằng {val}?", "Thử lũy thừa dần cơ số lên."],
        ask=f"{base} mũ 2 bằng mấy, mũ 3 bằng mấy... đến bao giờ bằng {val}?",
        practice=f"Tính log₂(32), log₃(243).")

def g_l12_tich_phan_so_phuc():
    kind = R.choice(["integral", "complex"])
    if kind == "integral":
        a = pick([2, 3]); hi = pick([1, 2, 3]); n = pick([3])  # ∫0^hi a x^2 dx = a*hi^3/3, với a=3 cho số nguyên
        a = 3  # đảm bảo nguyên
        val = a * hi ** 3 // 3
        return dict(problem=f"Tính tích phân I = ∫₀^{hi} {a}x² dx.",
            analysis="Tìm nguyên hàm của hàm dưới dấu tích phân rồi dùng công thức Newton-Leibniz: ∫(từ a đến b) f(x)dx = F(b) - F(a), trong đó F là nguyên hàm của f.",
            steps=[f"Nguyên hàm của {a}x² là {a}·x³/3 = x³", f"I = [x³] từ 0 đến {hi} = {hi}³ - 0³ = {val}"],
            answer=f"I = {val}",
            hints=["Nguyên hàm của x² là gì?", "Lấy nguyên hàm tại cận trên trừ tại cận dưới."],
            ask="Một nguyên hàm của 3x² là biểu thức nào?",
            practice="Tính ∫₀² (3x² + 2x) dx.")
    a1, b1, a2, b2 = ri(-5, 6), ri(-5, 6) or 2, ri(-5, 6), ri(-5, 6) or 3
    re_sum, im_sum = a1 + a2, b1 + b2
    # mô-đun của z1 chọn bộ Pythagoras để đẹp
    re_m, im_m = pick([(3, 4), (5, 12), (6, 8), (8, 15), (9, 12)])
    mod = int(math.hypot(re_m, im_m))
    return dict(problem=f"Cho hai số phức z₁ = {a1} {'+' if b1>=0 else '-'} {abs(b1)}i và z₂ = {a2} {'+' if b2>=0 else '-'} {abs(b2)}i. a) Tính z₁ + z₂. b) Tính mô-đun của số phức z = {re_m} + {im_m}i.",
        analysis="Cộng hai số phức: cộng phần thực với phần thực, phần ảo với phần ảo. Mô-đun của z = a + bi là |z| = √(a² + b²).",
        steps=[f"a) z₁ + z₂ = ({a1} + {a2}) + ({b1} + {b2})i = {re_sum} {'+' if im_sum>=0 else '-'} {abs(im_sum)}i",
               f"b) |z| = √({re_m}² + {im_m}²) = √({re_m*re_m} + {im_m*im_m}) = √{re_m*re_m+im_m*im_m} = {mod}"],
        answer=f"z₁ + z₂ = {re_sum} {'+' if im_sum>=0 else '-'} {abs(im_sum)}i; |z| = {mod}",
        hints=["Cộng số phức tách riêng phần thực và phần ảo.", "Mô-đun giống độ dài vectơ biểu diễn số phức."],
        ask="Phần thực của z₁ + z₂ bằng bao nhiêu?",
        practice="Cho z = 3 + 4i. Tính số phức liên hợp và mô-đun của z.")

def g_l12_oxyz_the_tich():
    if R.random() < 0.5:
        p1 = (ri(-4, 4), ri(-4, 4), ri(-4, 4))
        # chọn điểm mới cách theo bộ 3-4-? dùng vector (dx,dy,dz) sao cho khoảng cách nguyên: dùng (2,3,6)->7, (1,2,2)->3, (2,6,9 hmm) chọn list
        dv = pick([(1, 2, 2), (2, 3, 6), (2, 2, 1), (4, 4, 7), (3, 4, 0), (6, 8, 0)])
        p2 = tuple(p1[i] + dv[i] for i in range(3))
        dist = int(math.sqrt(sum(d * d for d in dv)))
        mid = tuple(Fraction(p1[i] + p2[i], 2) for i in range(3))
        return dict(problem=f"Trong không gian Oxyz, cho A{p1} và B{p2}. a) Tính độ dài đoạn AB. b) Tìm tọa độ trung điểm I của AB.",
            analysis="Khoảng cách AB = √[(xB-xA)² + (yB-yA)² + (zB-zA)²]. Trung điểm có từng tọa độ là trung bình cộng tọa độ tương ứng của A và B.",
            steps=[f"AB = √({dv[0]}² + {dv[1]}² + {dv[2]}²) = √{sum(d*d for d in dv)} = {dist}",
                   f"I = (({p1[0]}+{p2[0]})/2; ({p1[1]}+{p2[1]})/2; ({p1[2]}+{p2[2]})/2) = ({fmt(mid[0])}; {fmt(mid[1])}; {fmt(mid[2])})"],
            answer=f"AB = {dist}; I({fmt(mid[0])}; {fmt(mid[1])}; {fmt(mid[2])})",
            hints=["Công thức khoảng cách trong Oxyz mở rộng từ mặt phẳng thêm một tọa độ z.", "Trung điểm: lấy trung bình cộng từng tọa độ."],
            ask="Hiệu các tọa độ tương ứng của B và A là bao nhiêu?",
            practice="Cho M(1;-2;3), N(3;0;-1). Tính MN và trung điểm MN.")
    base_area = pick([6, 8, 9, 12, 15, 18]); h = pick([3, 4, 5, 6, 9])
    vol = Fraction(base_area * h, 3)
    return dict(problem=f"Một khối chóp có diện tích đáy bằng {base_area} (đơn vị diện tích) và chiều cao bằng {h} (đơn vị độ dài). Tính thể tích của khối chóp.",
        analysis="Thể tích khối chóp: V = (1/3) × diện tích đáy × chiều cao. (Chú ý phân biệt với khối lăng trụ V = B×h.)",
        steps=[f"V = (1/3) × {base_area} × {h}", f"= {fmt(vol)} (đơn vị thể tích)"],
        answer=f"V = {fmt(vol)}",
        hints=["Thể tích chóp chỉ bằng 1/3 thể tích lăng trụ cùng đáy và chiều cao.", "Đừng quên hệ số 1/3 nhé."],
        ask="Công thức thể tích khối chóp khác khối lăng trụ ở điểm nào?",
        practice=f"Khối chóp có đáy là hình vuông cạnh {ri(3,8)}, chiều cao {ri(4,9)}. Tính thể tích.")

def g_l12_cuc_tri_dao_ham():
    # Khảo sát cực trị hàm bậc 3 đơn giản: f(x)= x^3 - 3r^2 x -> f'=3x^2-3r^2 nghiệm ±r
    r = ri(1, 4); coeff = 3 * r * r
    f_at_min = r ** 3 - coeff * r; f_at_max = (-r) ** 3 - coeff * (-r)
    return dict(problem=f"Tìm các điểm cực trị của hàm số y = x³ - {coeff}x.",
        analysis="Tìm tập xác định, Tính y', giải y' = 0, lập bảng biến thiên xét dấu y', kết luận cực đại/cực tiểu dựa vào sự đổi dấu của đạo hàm.",
        steps=[f"TXĐ: D = ℝ. y' = 3x² - {coeff}", f"y' = 0 ⇔ x² = {r*r} ⇔ x = {-r} hoặc x = {r}",
               f"Bảng biến thiên: y' đổi dấu từ + sang - tại x = {-r} nên x = {-r} là điểm cực đại; đổi dấu từ - sang + tại x = {r} nên x = {r} là điểm cực tiểu",
               f"y_cực đại = y({-r}) = {f_at_max}, y_cực tiểu = y({r}) = {f_at_min}"],
        answer=f"Cực đại tại x = {-r} (y = {f_at_max}), cực tiểu tại x = {r} (y = {f_at_min})",
        hints=["Cực trị liên quan đến nghiệm của đạo hàm và sự đổi dấu của nó.", "Lập bảng biến thiên sẽ nhìn rõ cực đại/cực tiểu."],
        ask="y' bằng biểu thức nào và bằng 0 khi x bằng mấy?",
        practice="Tìm cực trị của y = x³ - 3x + 1.")

def g_lap_phuong_trinh():
    """Lớp 8-9: giải bài toán bằng cách lập phương trình (dạng năng suất/tổng)"""
    x = ri(12, 40); k = pick([2, 3]); total = x + k * x
    return dict(problem=f"Mẹ Lan gấp {k} lần tuổi Lan Tuổi hai mẹ con cộng lại bằng {total} tuổi. Hỏi mỗi người bao nhiêu tuổi? (Giải bằng cách lập phương trình)",
        analysis="Đặt ẩn là tuổi của Linh, biểu diễn tuổi mẹ qua ẩn, rồi lập phương trình từ tổng số tuổi.",
        steps=[f"Gọi tuổi của Linh là x (tuổi, x > 0). Tuổi mẹ là {k}x",
               f"Tổng số tuổi: x + {k}x = {total} ⇔ {k+1}x = {total} ⇔ x = {x} (thỏa mãn điều kiện)",
               f"Tuổi mẹ = {k} × {x} = {k*x} (tuổi)"],
        answer=f"Linh {x} tuổi, mẹ {k*x} tuổi",
        hints=["Nên đặt ẩn x là tuổi của người ít tuổi hơn.", "Viết mọi đại lượng còn lại theo x rồi mới lập phương trình."],
        ask="Nếu gọi tuổi Linh là x thì tuổi mẹ viết thế nào theo x?",
        practice="Tổng số tuổi của hai bố con là 52, tuổi bố gấp 3 lần tuổi con. Tính tuổi mỗi người.")

# Đăng ký (hàm, lớp, chủ đề, trọng số)
REGISTRY = [
    (g_l1_cong_tru, 1, "Cộng trừ trong phạm vi 100, bài toán lời văn", 3),
    (g_l1_so_sanh, 1, "So sánh số", 2),
    (g_l2_nhan_chia, 2, "Bảng nhân, bảng chia", 3),
    (g_l2_cong_tru_1000, 2, "Cộng trừ trong phạm vi 1000", 2),
    (g_l3_chuvi_dientich, 3, "Chu vi, diện tích", 3),
    (g_l4_phan_so_cong, 4, "Phân số: cộng, quy đồng", 3),
    (g_l4_trung_binh_cong_tong_hieu, 4, "Trung bình cộng; tổng-hiệu", 3),
    (g_l5_phan_tram, 5, "Tỉ số phần trăm", 3),
    (g_l5_van_toc, 5, "Vận tốc, quãng đường, thời gian", 3),
    (g_l5_the_tich, 5, "Thể tích hình hộp chữ nhật", 2),
    (g_l6_so_nguyen_ucln, 6, "Số nguyên; ƯCLN-BCNN", 3),
    (g_l6_phan_so_nhan_chia, 6, "Phân số: nhân, chia", 3),
    (g_l7_ti_le_thuc_da_thuc, 7, "Tỉ lệ thức; giá trị đa thức", 3),
    (g_l7_luy_thua_goc, 7, "Luỹ thừa; góc kề bù", 3),
    (g_l8_hang_dang_thuc_pt, 8, "Hằng đẳng thức; PT bậc nhất; phân tích đa thức", 4),
    (g_l8_bat_phuong_trinh, 8, "Bất phương trình bậc nhất", 2),
    (g_lap_phuong_trinh, 8, "Giải bài toán bằng lập phương trình", 2),
    (g_l9_can_he_bac2, 9, "Căn; hệ phương trình; PT bậc hai Viète", 4),
    (g_l9_ham_so_luong_giac_vuong, 9, "Hàm bậc nhất; tỉ số lượng giác", 3),
    (g_l10_vecto_bpt2, 10, "Vectơ tọa độ; BPT bậc hai; lượng giác cơ bản", 4),
    (g_l10_duong_thang, 10, "Phương trình đường thẳng", 2),
    (g_l11_capso_tohop, 11, "Cấp số cộng/nhân; tổ hợp", 3),
    (g_l11_xac_suat_dao_ham, 11, "Xác suất cổ điển; đạo hàm cơ bản", 3),
    (g_l11_gioi_han, 11, "Giới hạn dạng 0/0", 2),
    (g_l12_mu_log, 12, "Phương trình mũ; logarit", 3),
    (g_l12_tich_phan_so_phuc, 12, "Tích phân cơ bản; số phức", 3),
    (g_l12_oxyz_the_tich, 12, "Oxyz khoảng cách; thể tích khối chóp", 3),
    (g_l12_cuc_tri_dao_ham, 12, "Cực trị hàm bậc ba", 3),
]

USER_TEMPLATES_DETAILED = [
    "{p}", "Giải chi tiết bài Toán sau giúp em:\n{p}", "Em chưa hiểu bài này, hướng dẫn giải từng bước giúp em:\n{p}",
    "Giải chi tiết (lớp {grade}): {p}", "Bài Toán: {p}\nHãy trình bày lời giải chi tiết.",
]
USER_TEMPLATES_HINT = [
    "Gợi ý cho em bài này, đừng cho em đáp án ngay:\n{p}", "Chế độ gợi ý: {p}",
    "Em muốn tự làm, thầy/cô gợi ý cho em nhé:\n{p}", "Chỉ gợi ý cách làm bài này thôi ạ:\n{p}",
]

def build_example(item, grade, mode):
    p = item["problem"]
    if mode == "giai_chi_tiet":
        user = pick(USER_TEMPLATES_DETAILED).format(p=p, grade=grade)
        steps_txt = "\n".join(f"- Bước {i+1}: {s}" for i, s in enumerate(item["steps"]))
        assistant = (f"📐 **Phân tích đề:** {item['analysis']}\n\n**Lời giải từng bước:**\n{steps_txt}\n\n"
                     f"✅ **Đáp án:** {item['answer']}\n\n✏️ **Bài luyện tương tự (em tự làm nhé):** {item['practice']}")
    else:
        user = pick(USER_TEMPLATES_HINT).format(p=p)
        hints_txt = "\n".join(f"- **Gợi ý {i+1}:** {h}" for i, h in enumerate(item["hints"]))
        assistant = (f"Rất tốt, em hãy tự suy luận nhé! 💪 Thầy/cô sẽ không cho đáp án ngay đâu.\n\n{hints_txt}\n\n"
                     f"❓ **Câu hỏi cho em:** {item['ask']}\n\nEm làm được bước đó rồi nhắn lại, mình đi tiếp bước sau nhé!")
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user},
                          {"role": "assistant", "content": assistant}],
            "grade": grade, "mode": mode}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    global R
    R = random.Random(args.seed)

    weights = [w for _, _, _, w in REGISTRY]
    seen = set(); examples = []
    tries = 0
    while len(examples) < args.n and tries < args.n * 30:
        tries += 1
        fn, grade, topic, _ = R.choices(REGISTRY, weights=weights, k=1)[0]
        try:
            item = fn()
        except Exception as e:
            continue
        mode = "giai_chi_tiet" if R.random() < 0.65 else "goi_y"
        ex = build_example(item, grade, mode)
        ex["topic"] = topic
        key = ex["messages"][1]["content"]
        if key in seen:
            continue
        seen.add(key)
        examples.append(ex)

    R.shuffle(examples)
    n = len(examples)
    val, train = examples[:max(50, n // 20)], examples[max(50, n // 20):]
    os.makedirs(args.out_dir, exist_ok=True)
    for name, part in [("train.jsonl", train), ("val.jsonl", val)]:
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
            for ex in part:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Thống kê
    from collections import Counter
    cg = Counter(e["grade"] for e in examples); cm = Counter(e["mode"] for e in examples)
    stats = {"tong_mau": n, "train": len(train), "val": len(val),
             "theo_lop": {str(k): cg[k] for k in sorted(cg)}, "theo_che_do": dict(cm)}
    with open(os.path.join(args.out_dir, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\n--- Ví dụ mẫu ---\n", train[0]["messages"][1]["content"], "\n---\n", train[0]["messages"][2]["content"])

if __name__ == "__main__":
    main()

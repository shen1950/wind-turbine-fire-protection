from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def _font(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def save_metric_chart(metrics, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    W, H = 1400, 760
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    title_font = _font(38, True)
    label_font = _font(22)
    small_font = _font(18)
    d.text((70, 35), "火灾识别算法仿真性能对比", fill="#17365D", font=title_font)
    d.line((70, 92, 1320, 92), fill="#6D9EBD", width=4)
    x0, y0, x1, y1 = 120, 610, 1280, 140
    d.line((x0, y0, x1, y0), fill="#333333", width=3)
    d.line((x0, y0, x0, y1), fill="#333333", width=3)
    for i in range(7):
        val = 70 + i * 5
        y = y0 - (val - 70) / 30 * (y0 - y1)
        d.line((x0 - 8, y, x1, y), fill="#E8E8E8", width=1)
        d.text((54, y - 12), f"{val}%", fill="#555555", font=small_font)

    names = [m["model"] for m in metrics]
    acc = [m["accuracy"] * 100 for m in metrics]
    anti_false = [(1 - m["false_alarm_rate"]) * 100 for m in metrics]
    group_gap = 205
    bar_w = 55
    start = 210
    for i, name in enumerate(names):
        gx = start + i * group_gap
        for j, (value, color) in enumerate([(acc[i], "#2E74B5"), (anti_false[i], "#70AD47")]):
            bx = gx + j * (bar_w + 12)
            shown = max(70, min(100, value))
            by = y0 - (shown - 70) / 30 * (y0 - y1)
            d.rectangle((bx, by, bx + bar_w, y0), fill=color)
            d.text((bx - 6, by - 26), f"{value:.1f}", fill="#333333", font=small_font)
        d.text((gx - 30, y0 + 22), name, fill="#333333", font=small_font)

    d.rectangle((880, 52, 910, 82), fill="#2E74B5")
    d.text((920, 53), "准确率", fill="#333333", font=label_font)
    d.rectangle((1035, 52, 1065, 82), fill="#70AD47")
    d.text((1075, 53), "100%-误报率", fill="#333333", font=label_font)
    d.text((120, 670), "说明：数据为课程设计仿真样本输出，运行 run_all.py 可重新生成。", fill="#555555", font=small_font)
    img.save(out_path, quality=95)

"""Relabel the preserved Exp45 error figure without touching scientific panels."""

from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "qualitative_results" / "qualitative_results_exp45_errors.png"
BACKUP = TARGET.with_name("qualitative_results_exp45_errors_before_label_update.png")
HIGH_RES_COPY = TARGET.with_name("qualitative_results_exp45_errors_600dpi.png")
SCALE = 1.5


def centered_text(draw, box, text, font):
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x = left + ((right - left) - width) / 2 - bounds[0]
    y = top + ((bottom - top) - height) / 2 - bounds[1]
    draw.text((x, y), text, fill=(0, 0, 0, 255), font=font)


def main():
    if not BACKUP.exists():
        BACKUP.write_bytes(TARGET.read_bytes())

    with Image.open(BACKUP) as source:
        image = source.convert("RGBA")

    image = image.resize(
        (round(image.width * SCALE), round(image.height * SCALE)),
        Image.Resampling.LANCZOS,
    )

    draw = ImageDraw.Draw(image)
    bold_path = font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans", weight="bold")
    )
    header_font = ImageFont.truetype(bold_path, 133)
    row_font = ImageFont.truetype(bold_path, 108)
    legend_font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 88)

    def scaled(box):
        return tuple(round(value * SCALE) for value in box)

    # Clear only the original typography margins; panels begin below/right.
    header_box = scaled((400, 0, 4800, 126))
    row_label_box = scaled((240, 126, 400, 3720))
    legend_box = scaled((1400, 3627, 3400, 3720))
    draw.rectangle(header_box, fill=(255, 255, 255, 255))
    draw.rectangle(row_label_box, fill=(255, 255, 255, 255))
    draw.rectangle(legend_box, fill=(255, 255, 255, 255))

    column_centers = (718, 1350, 1982, 2614, 3246, 3878, 4510)
    column_labels = ("Input", "GT", "Baseline", "CTNet", "MEGANet", "EnFormer", "Ours")
    for label, center_x in zip(column_labels, column_centers):
        centered_text(
            draw,
            scaled((center_x - 300, 0, center_x + 300, 126)),
            label,
            header_font,
        )

    row_centers = (415, 993, 1570, 2147, 2724, 3302)
    for label, center_y in zip("ABCDEF", row_centers):
        centered_text(draw, scaled((240, center_y - 90, 400, center_y + 90)), label, row_font)

    centered_text(
        draw,
        legend_box,
        "Error colors: white = TP, red = FP, cyan = FN",
        legend_font,
    )

    output_dpi = (600, 600)
    image.save(TARGET, dpi=output_dpi)
    image.save(HIGH_RES_COPY, dpi=output_dpi)
    print(f"Saved {TARGET} at 600 DPI ({image.width} x {image.height})")
    print(f"Saved {HIGH_RES_COPY}")
    print(f"Backup {BACKUP}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    base = Path("outputs/tw1_minpa_quality_flag_audit")
    figdir = base / "examples"
    examples_csv = base / "tw1_minpa_hplus_quality_flag_audit_examples.csv"
    rows = list(csv.DictReader(examples_csv.open(newline="")))

    thumbs: list[Image.Image] = []
    for row in rows:
        time_tag = (
            row["time_utc"]
            .replace("-", "")
            .replace(":", "")
            .replace("T", "_")
            .replace("Z", "")
        )
        time_tag = time_tag[:15] + time_tag[16:19]
        image_path = figdir / f"{row['category']}_{time_tag}_flag{row['quality_flag']}.png"
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((420, 245))
        canvas = Image.new("RGB", (420, 275), "white")
        canvas.paste(image, ((420 - image.width) // 2, 25))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 6),
            f"{row['category']}  bits={row['quality_flag_binary']}",
            fill=(0, 0, 0),
        )
        thumbs.append(canvas)

    cols = 2
    rows_n = (len(thumbs) + cols - 1) // cols
    output = Image.new("RGB", (cols * 420, rows_n * 275), "white")
    for idx, thumb in enumerate(thumbs):
        output.paste(thumb, ((idx % cols) * 420, (idx // cols) * 275))

    out_path = base / "tw1_minpa_quality_flag_examples_montage.png"
    output.save(out_path)
    print(out_path.resolve())


if __name__ == "__main__":
    main()

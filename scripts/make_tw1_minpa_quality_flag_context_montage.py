from __future__ import annotations

import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    if len(sys.argv) > 1:
        base = Path(sys.argv[1])
    else:
        base = Path("outputs/tw1_minpa_quality_flag_audit/context_5min_project_spectrogram")
    index_csv = base / "tw1_minpa_quality_flag_context_5min_index.csv"
    rows = list(csv.DictReader(index_csv.open(newline="")))

    thumbs: list[Image.Image] = []
    for row in rows:
        png = Path(row["png"])
        if not png.exists():
            continue
        image = Image.open(png).convert("RGB")
        image.thumbnail((520, 360))
        canvas = Image.new("RGB", (520, 390), "white")
        canvas.paste(image, ((520 - image.width) // 2, 28))
        draw = ImageDraw.Draw(canvas)
        label = f"{row['category']}  {row['time_utc']}  bits={row['quality_flag_binary']}"
        draw.text((8, 7), label, fill=(0, 0, 0))
        thumbs.append(canvas)

    cols = 2
    rows_n = (len(thumbs) + cols - 1) // cols
    output = Image.new("RGB", (cols * 520, rows_n * 390), "white")
    for idx, thumb in enumerate(thumbs):
        output.paste(thumb, ((idx % cols) * 520, (idx // cols) * 390))

    out_path = base / "tw1_minpa_quality_flag_context_5min_project_spectrogram_montage.png"
    output.save(out_path)
    print(out_path.resolve())


if __name__ == "__main__":
    main()

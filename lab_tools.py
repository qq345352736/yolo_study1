"""Helpers for the public Open Images traffic workshop."""
from pathlib import Path
from collections import Counter
import csv
import random
import shutil

import yaml
from PIL import Image, ImageDraw, ImageFont

# BASE = Path(__file__).resolve().parent
# NAMES = (BASE / "classes.txt").read_text(encoding="utf-8-sig").splitlines()
# RAW = BASE / "dataset"

BASE=Path("F:/work_new/QingMiaoChunLei/2026-11/Notes/FangZiQuan/11月24日课程材料/上午材料/实验二 数据整理")
NAMES = (BASE / "classes.txt").read_text(encoding="utf-8-sig").splitlines()
RAW = BASE / "dataset"

# NAMES="F:/work_new/QingMiaoChunLei/2026-11/Notes/FangZiQuan/11月24日课程材料/上午材料/实验二 数据整理/classes.txt"
# RAW="F:/work_new/QingMiaoChunLei/2026-11/Notes/FangZiQuan/11月24日课程材料/上午材料/实验二 数据整理/dataset"

COLORS = [(20, 129, 182), (197, 72, 18), (32, 140, 67)]


def rows():
    with (RAW / "meta.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def labels(filename):
    path = RAW / "labels" / Path(filename).with_suffix(".txt")
    return [[float(v) for v in line.split()] for line in path.read_text().splitlines() if line.strip()]


def counts(records):
    return Counter(NAMES[int(box[0])] for row in records for box in labels(row["filename"]))


def prepare(seed=0):
    """Split photographer accounts into train/val/test and write the local YOLO config."""
    records = rows()
    groups = sorted({r["group"] for r in records})
    random.Random(seed).shuffle(groups)
    train_end = round(len(groups) * 0.7)
    val_end = round(len(groups) * 0.9)
    part = {group: ("train" if i < train_end else "val" if i < val_end else "test")
            for i, group in enumerate(groups)}
    dest = BASE / "划分结果"
    for split in ["train", "val", "test"]:
        subset = [r for r in records if part[r["group"]] == split]
        assert set(counts(subset)) == set(NAMES), f"Missing class in {split}"
        for subdir, ext in [("images", ".jpg"), ("labels", ".txt")]:
            target = dest / subdir / split
            target.mkdir(parents=True, exist_ok=True)
            expected = {str(Path(r["filename"]).with_suffix(ext)) for r in subset}
            if set(p.name for p in target.glob("*" + ext)) - expected:
                raise ValueError(f"Old split in {target}. Use an empty output folder before changing the seed.")
            for name in expected:
                shutil.copyfile(RAW / subdir / name, target / name)
        print(f"{split:5}   {len(subset):3} images   {len({r['group'] for r in subset}):3} photographers")
    with (dest / "split.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "group", "split"])
        writer.writerows((r["filename"], r["group"], part[r["group"]]) for r in records)
    config = {"path": dest.resolve().as_posix(), "train": "images/train", "val": "images/val",
              "test": "images/test", "names": dict(enumerate(NAMES))}
    (dest / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return dest


def annotated(filename):
    with Image.open(RAW / "images" / filename) as source:
        im = source.convert("RGB")
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    for cls, x, y, w, h in labels(filename):
        color = COLORS[int(cls)]
        left, top = (x-w/2)*im.width, (y-h/2)*im.height
        right, bottom = (x+w/2)*im.width, (y+h/2)*im.height
        draw.rectangle((left, top, right, bottom), outline=color, width=4)
        draw.text((left+2, max(1, top-13)), NAMES[int(cls)], fill="white", stroke_width=2, stroke_fill=color, font=font)
    return im


def contact_sheet(images, cols=3, tile=(245, 152)):
    images = list(images)
    width, height = tile
    sheet = Image.new("RGB", (width * cols, height*((len(images)+cols-1)//cols)), "white")
    for i, im in enumerate(images):
        im = im.copy()
        im.thumbnail((width-4, height-4))
        x = (i%cols)*width+(width-im.width)//2
        y = (i//cols)*height+(height-im.height)//2
        sheet.paste(im, (x,y))
    return sheet


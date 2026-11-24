from pathlib import Path
import random
from IPython.display import display
from lab_tools import rows, counts, prepare, annotated, contact_sheet

# 每行 YOLO 标注对应一个框。一张街景里可能有多辆车。
records = rows()
print(f"Photos: {len(records)}    Boxes: {sum(counts(records).values())}")
for name, n in counts(records).most_common():
    print(f"{name:8} {n:3} boxes")

sample = random.Random(0).sample(records, 9)
display(contact_sheet([annotated(r["filename"]) for r in sample]))

# 按摄影者划分
# 以摄影者账号为组，按约 7∶2∶1 分配组。每组照片数量不同，图片比例会略有偏差。

dataset = prepare(seed=0)


print((dataset / "data.yaml").read_text(encoding="utf-8"))
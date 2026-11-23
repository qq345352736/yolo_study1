## 测试Python代码
print("Hello, World!")


from ultralytics import YOLO
model = YOLO("yolo26n.pt")   # 首次运行自动下载权重
jpgs = ["F:\\work_new\\QingMiaoChunLei\\2026-11\\Notes\\FangZiQuan\\Codes\\上午材料\\lab1_images\\01.jpg",
        "F:\\work_new\\QingMiaoChunLei\\2026-11\\Notes\\FangZiQuan\\Codes\\上午材料\\lab1_images\\02.jpg",
        "F:\\work_new\\QingMiaoChunLei\\2026-11\\Notes\\FangZiQuan\\Codes\\上午材料\\lab1_images\\03.jpg",
        "F:\\work_new\\QingMiaoChunLei\\2026-11\\Notes\\FangZiQuan\\Codes\\上午材料\\lab1_images\\04.jpg",
        "F:\\work_new\\QingMiaoChunLei\\2026-11\\Notes\\FangZiQuan\\Codes\\上午材料\\lab1_images\\07.jpg",
        "F:\\work_new\\QingMiaoChunLei\\2026-11\\Notes\\FangZiQuan\\Codes\\上午材料\\lab1_images\\08.jpg",
        "F:\\work_new\\QingMiaoChunLei\\2026-11\\Notes\\FangZiQuan\\Codes\\上午材料\\lab1_images\\09.jpg",
        "F:\\work_new\\QingMiaoChunLei\\2026-11\\Notes\\FangZiQuan\\Codes\\上午材料\\lab1_images\\11.jpg",
        "F:\\work_new\\QingMiaoChunLei\\2026-11\\Notes\\FangZiQuan\\Codes\\上午材料\\lab1_images\\14.jpg",
        "F:\\work_new\\QingMiaoChunLei\\2026-11\\Notes\\FangZiQuan\\Codes\\上午材料\\lab1_images\\18.jpg"]

results = model(
    source=jpgs,
    project="F:\\work_new\\QingMiaoChunLei\\2026-11\\Notes\\FangZiQuan\\Codes\\yolo_output",
    save=True)
print("YOLO OK")
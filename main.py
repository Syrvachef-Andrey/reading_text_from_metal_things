import os

# 1. Железобетонно отключаем багнутый ускоритель на уровне ОС ДО импорта Paddle
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import cv2
from paddleocr import PaddleOCR

# 2. Инициализируем OCR с явным запретом MKL-DNN
ocr = PaddleOCR(
    det_model_dir='./models/Multilingual_PP-OCRv3_det_infer',
    rec_model_dir='./models/cyrillic_PP-OCRv3_rec_infer',
    cls_model_dir='./models/ch_ppocr_mobile_v2.0_cls_infer',
    use_angle_cls=True,
    lang='ru',
    show_log=False,
    use_mkldnn=False,     # КРИТИЧЕСКИ ВАЖНО
    enable_mkldnn=False   # КРИТИЧЕСКИ ВАЖНО
)

img_path = "image_for_test/image.png"

print("Начинаю распознавание из локальных файлов...")

# Выполняем распознавание
result = ocr.ocr(img_path, cls=True)

# Оцениваем и выводим результат
print("\n--- РЕЗУЛЬТАТЫ ---")
if result and result[0]:
    for idx, line in enumerate(result[0]):
        box = line[0]
        text = line[1][0]
        confidence = line[1][1]

        print(f"[{idx + 1}] Текст: {text:<25} | Уверенность: {confidence:.3f}")
else:
    print("Текст на изображении не найден.")
print("-------------------")
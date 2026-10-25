import cv2
import numpy as np
import torch
import easyocr
from PIL import Image, ImageDraw, ImageFont
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os
import configparser

my_config = configparser.ConfigParser()
my_config.read("./config.ini")

os.environ["HF_TOKEN"] = my_config["HF_TOKEN"]["token"]

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

detector = easyocr.Reader(["en"], gpu=False)

opened_image = cv2.imread(my_config["PATH"]["image"])

pts_src = np.array([[549, 234], [1050, 226], [1064, 532], [543, 512]], dtype=np.float32)
(tl, tr, br, bl) = pts_src
widthA = np.linalg.norm(br - bl)
widthB = np.linalg.norm(tr - tl)
width = max(int(widthA), int(widthB))
heightA = np.linalg.norm(tr - br)
heightB = np.linalg.norm(tl - bl)
height = max(int(heightA), int(heightB))

# 1. Глобальные отступы для всего шильдика
PLATE_PAD = 20 # Запас в 20 пикселей со всех сторон
pts_dst = np.array([
    [PLATE_PAD, PLATE_PAD],
    [width + PLATE_PAD, PLATE_PAD],
    [width + PLATE_PAD, height + PLATE_PAD],
    [PLATE_PAD, height + PLATE_PAD]
], dtype=np.float32)

matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
# Увеличиваем итоговый холст, чтобы вместить отступы
warped_image = cv2.warpPerspective(opened_image, matrix, (width + 2 * PLATE_PAD, height + 2 * PLATE_PAD))

img_pil = Image.fromarray(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(img_pil)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
except IOError:
    font = ImageFont.load_default()

detection_results = detector.readtext(warped_image)
img_h, img_w = warped_image.shape[:2]

# 2. Локальные отступы для каждого слова
WORD_PAD = 8 # Запас вокруг слова для TrOCR

for idx, (bbox, _, _) in enumerate(detection_results):
    pts = np.array(bbox, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)

    # Расширяем рамку вырезания, проверяя границы изображения
    x1 = max(0, x - WORD_PAD)
    y1 = max(0, y - WORD_PAD)
    x2 = min(img_w, x + w + WORD_PAD)
    y2 = min(img_h, y + h + WORD_PAD)

    crop_img = warped_image[y1:y2, x1:x2]

    if crop_img.size == 0:
        continue

    crop_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    pixel_values = processor(crop_pil, return_tensors="pt").pixel_values

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"[{idx + 1}] Найдено: {generated_text}")

    # Отрисовка рамки по оригинальным координатам для визуальной аккуратности
    draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
    draw.text((x, y - 20), generated_text, font=font, fill="blue")

final_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
# Убедись, что папка существует, иначе cv2.imwrite не сработает
os.makedirs("./output/tr_ocr_output", exist_ok=True)
cv2.imwrite("./output/tr_ocr_output/tr_ocr.png", final_img)
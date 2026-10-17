import cv2
import pytesseract
from pytesseract import Output
import numpy as np
from PIL import Image, ImageDraw, ImageFont

opened_image = cv2.imread('image_for_test/image.png')

pts_src = np.array([
    [549, 234],
    [1050, 226],
    [1064, 532],
    [543, 512]
], dtype=np.float32)

width, height = 350, 200

pts_dst = np.array([
    [0, 0],
    [width, 0],
    [width, height],
    [0, height]
], dtype=np.float32)

matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
warped_image = cv2.warpPerspective(opened_image, matrix, (width, height))

gray_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)

tess_data = pytesseract.image_to_data(binary_image, lang='rus+eng', config='--psm 6', output_type=Output.DICT)

img_pil = Image.fromarray(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(img_pil)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
except IOError:
    font = ImageFont.load_default()

n_boxes = len(tess_data['text'])
for i in range(n_boxes):
    conf = int(tess_data['conf'][i])
    text = tess_data['text'][i].strip()

    if conf > 10 and text:
        x, y, w, h = tess_data['left'][i], tess_data['top'][i], tess_data['width'][i], tess_data['height'][i]

        draw.rectangle([x, y, x + w, y + h], outline=(0, 0, 255), width=2)

        label = f"{text} ({conf}%)"
        draw.text((x, y - 18), label, font=font, fill=(255, 0, 0))

final_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

cv2.imwrite("./output/tesseract_out/tesseract_output.png", final_img)
cv2.imwrite("./output/tesseract_out/tesseract_binary_input.png", binary_image)
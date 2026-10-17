from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_textline_orientation=True,
    lang='ru',
    enable_mkldnn=False
)

img_path = 'image_for_test/image.png'
result = ocr.predict(img_path)

for res in result:
    res.save_to_img("./output/paddle_ocr_out/")

    for text, score in zip(res['rec_texts'], res['rec_scores']):
        print(f"{text}  (conf: {score:.3f})")
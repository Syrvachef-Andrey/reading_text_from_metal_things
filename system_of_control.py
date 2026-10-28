import os
import time
import cv2
import json
import numpy as np
import pandas as pd
import Levenshtein
import torch
import configparser
import pytesseract
import easyocr
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image


my_config = configparser.ConfigParser()
my_config.read("./config.ini")

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["HF_TOKEN"] = my_config["HF_TOKEN"]["token"]


def rotate_continuous_gaussian(image, mean_angle=0.0, std_angle=15.0):
    angle = np.random.normal(loc=mean_angle, scale=std_angle)

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    cos_val = np.abs(M[0, 0])
    sin_val = np.abs(M[0, 1])
    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated

def simulate_camera_perspective_tilt(image, max_tilt_ratio=0.15):
    h, w = image.shape[:2]

    pts_src = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ])

    std_offset = min(w, h) * max_tilt_ratio / 3.0
    offsets = np.random.normal(0, std_offset, (4, 2)).astype(np.float32)
    pts_dst = pts_src + offsets

    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    tilted = cv2.warpPerspective(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return tilted

def add_noise(image):
    row, col, ch = image.shape
    mean, sigma = 0, 25
    gauss = np.random.normal(mean, sigma, (row, col, ch)).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
    return noisy

def add_glare(image):
    glare = np.zeros_like(image, dtype=np.uint8)
    h, w = image.shape[:2]
    cv2.ellipse(glare, (w // 2, h // 2), (w // 3, h // 4), 30, 0, 360, (255, 255, 255), -1)
    glare = cv2.GaussianBlur(glare, (51, 51), 0)
    return cv2.addWeighted(image, 0.7, glare, 0.4, 0)

def calculate_cer(ground_truth, predicted):
    gt_clean = " ".join(ground_truth.upper().split())
    pred_clean = " ".join(predicted.upper().split())
    if not gt_clean: return 100.0
    dist = Levenshtein.distance(gt_clean, pred_clean)
    return min(100.0, round((dist / len(gt_clean)) * 100, 2))

def run_tesseract(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    return pytesseract.image_to_string(binary, lang='rus+eng', config='--psm 6').strip()

def run_paddle(img, paddle_model):
    res = paddle_model.ocr(img, cls=False)
    if not res or not res[0]: return ""
    return " ".join([line[1][0] for line in res[0]])

def run_easyocr(img, easy_model):
    res = easy_model.readtext(img)
    return " ".join([text for _, text, _ in res])

def run_trocr(img, easy_model, processor, trocr_model):
    det_results = easy_model.readtext(img)
    h_img, w_img = img.shape[:2]
    texts = []

    for bbox, _, _ in det_results:
        pts = np.array(bbox, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)

        pad = 8
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0: continue

        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pixel_values = processor(crop_pil, return_tensors="pt").pixel_values
        with torch.no_grad():
            gen_ids = trocr_model.generate(pixel_values)
        texts.append(processor.batch_decode(gen_ids, skip_special_tokens=True)[0])

    return " ".join(texts)


def main():
    print("Инициализация моделей")
    paddle_m = PaddleOCR(lang='ru', use_angle_cls=False, show_log=False, use_mkldnn=False, enable_mkldnn=False)
    easy_m = easyocr.Reader(['ru', 'en'], gpu=False)
    trocr_proc = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    trocr_m = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

    dataset_dir = my_config["PATH"]["path_to_dataset"]
    gt_file = my_config["PATH"]["path_to_json_file"]

    if not os.path.exists(gt_file):
        print("Файл ground_truth.json не найден")
        ground_truth_dict = {"image.png": "ЭТАЛОННЫЙ ТЕКСТ ШИЛЬДИКА"}
        dataset_dir = my_config["PATH"]["path_to_test_image"]
    else:
        with open(gt_file, 'r', encoding='utf-8') as f:
            ground_truth_dict = json.load(f)

    results_list = []

    print("\nСтарт тестирования")
    for filename, true_text in ground_truth_dict.items():
        img_path = os.path.join(dataset_dir, filename)
        if not os.path.exists(img_path): continue

        orig_img = cv2.imread(img_path)

        images_to_test = {
            "Original": orig_img,
            "Noise": add_noise(orig_img),
            "Glare": add_glare(orig_img),
            "Rotate_Jitter": rotate_continuous_gaussian(orig_img, mean_angle=0.0, std_angle=15.0),
            "Persepctive_Tilt": simulate_camera_perspective_tilt(orig_img, max_tilt_ratio=0.12)
        }

        for defect_name, img in images_to_test.items():
            print(f"Обработка: {filename} | Состояние: {defect_name}")

            models = {
                "Tesseract": lambda: run_tesseract(img),
                "EasyOCR": lambda: run_easyocr(img, easy_m),
                "PaddleOCR": lambda: run_paddle(img, paddle_m),
                "TrOCR": lambda: run_trocr(img, easy_m, trocr_proc, trocr_m)
            }

            for model_name, run_func in models.items():
                t0 = time.time()
                try:
                    pred_text = run_func()
                except cv2.error as e:
                    # Ошибка OpenCV
                    pred_text = "CV2_ERROR"
                    print(f"Ошибка CV2 ({model_name} - {filename}): {e}")
                except RuntimeError as e:
                    # Ошибки PyTorch/Paddle (
                    pred_text = "TENSOR_RUNTIME_ERROR"
                    print(f"Ошибка вычислений ({model_name} - {filename}): {e}")
                except pytesseract.TesseractError as e:
                    # Специфичная ошибка движка Tesseract
                    pred_text = "TESSERACT_ENGINE_ERROR"
                    print(f"Ошибка Tesseract ({model_name} - {filename}): {e}")
                except ValueError as e:
                    # Детекция ничего не нашла
                    pred_text = "VALUE_ERROR"
                    print(f"Ошибка значений ({model_name} - {filename}): {e}")
                except Exception as e:
                    pred_text = "UNKNOWN_CRASH"
                    print("FATAL UNKNOWN ERROR")

                inference_time = time.time() - t0

                cer_score = calculate_cer(true_text, pred_text)

                results_list.append({
                    "Image": filename,
                    "Defect": defect_name,
                    "Model": model_name,
                    "Ground Truth": true_text,
                    "Predicted": pred_text,
                    "CER (%)": cer_score,
                    "Time (s)": round(inference_time, 2)
                })

    if not results_list:
        print("Ни одно изображение не обработано таблица пуста")
        return

    df = pd.DataFrame(results_list)
    df.to_csv("ocr_benchmark_results.csv", index=False, encoding='utf-8-sig')

    summary = df.groupby(["Model", "Defect"])[["CER (%)", "Time (s)"]].mean().reset_index()
    print(summary.to_string())
    print("\n Результаты сохранены в 'ocr_benchmark_results.csv'")


if __name__ == "__main__":
    main()
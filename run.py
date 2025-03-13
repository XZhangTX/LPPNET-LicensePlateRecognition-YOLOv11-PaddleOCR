from ultralytics import YOLO
import os
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Load a model
model = YOLO("runs/detect/train13/weights/last.pt")  # pretrained YOLO model
# Load an OCR model into memory
ocr = PaddleOCR(lang='ch')
font = ImageFont.truetype("tools/STSONG.TTF", 70)

# Define the input and output directories
input_dir = "car_plate/test/img"
output_dir = "car_plate/results"
os.makedirs(output_dir, exist_ok=True)

images = [os.path.join(input_dir, img) for img in os.listdir(input_dir) if
          img.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Process each image
for image_path in images:
    result = model(image_path)[0]  # Perform object detection
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    for idx, box in enumerate(result.boxes.xyxy):  # Iterate over detected objects
        x1, y1, x2, y2 = map(int, box[:4])
        cropped_img = img.crop((x1, y1, x2, y2))

        # Convert the cropped image to a numpy array
        cropped_img_np = np.asarray(cropped_img)

        # Perform OCR on the cropped image
        ocr_result = ocr.ocr(cropped_img_np, det=False, cls=False)

        # Draw bounding box and text
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
        # try:
        #     font = ImageFont.truetype(.ttf", 50)  # load a font
        # except IOError:
        #     font = ImageFont.load_default()
        for line_idx, res in enumerate(ocr_result[0]):  # assume only one line of text
            text, score = res  # 解包元组
            print(f"Detected text: {text}, Score: {score}")
            draw.text((x1, y1 + line_idx * 60), str(text), fill="red", font=font)  # adjust the position of the text

        # Save the processed image
        output_path = os.path.join(output_dir, f"{os.path.basename(image_path)}_plate_{idx}.jpg")
        img.save(output_path)
        print(f"Processed and saved with OCR results: {output_path}")

# server_node.py
import io

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50

app = FastAPI()
# DeepLabV3 + ResNet50（COCOデータセットで学習済み）を読み込み 推論モードに設定
model = deeplabv3_resnet50(pretrained=True).eval()
# 画像前処理パイプライン
transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# フォームデータとして画像ファイル(UploadFile)を受け取る
@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
        preds = output.argmax(0).byte().cpu().numpy()

    mask = (preds == 15).astype(np.uint8) * 255
    np_img = np.array(img)
    overlay = np_img.copy()
    overlay[mask == 255] = [0, 0, 255]
    blended = cv2.addWeighted(np_img, 0.6, overlay, 0.4, 0)

    _, jpeg = cv2.imencode(".jpg", blended)
    return StreamingResponse(io.BytesIO(jpeg.tobytes()), media_type="image/jpeg")

if __name__ == "__main__":
    # uvicorn server_node:app --host 0.0.0.0 --port 8080
    uvicorn.run(app, host="0.0.0.0", port=8000)

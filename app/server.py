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
from fastapi.responses import FileResponse
from pathlib import Path
import os
from fastapi import HTTPException

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

@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    """
    画像を受け取ると、人を検出して色のマスクを重ねた画像を返す
    """
    # アップロード画像を読み込む
    img = Image.open(file.file).convert("RGB")
    print('処理前',img.size)
    # 前処理を適用
    input_tensor = transform(img)
    input_tensor = input_tensor.unsqueeze(0)  # (1, C, H, W)
    print(type(input_tensor))

    # モデルで推論 既に推論モード
    with torch.no_grad():
        output = model(input_tensor)
        output = output['out'][0]  # (num_classes, H, W)
        preds = output.argmax(0).byte().cpu().numpy()  # 各ピクセルのクラスID

    # マスク生成（クラス15=person）
    mask = (preds == 15).astype(np.uint8) * 255
    # 元画像にオーバーレイ
    np_img = np.array(img)  # 元画像サイズのまま
    h0, w0 = np_img.shape[:2]
    # mask を元画像サイズに拡大/縮小,最近傍補間（INTER_NEAREST）を使うことでクラス境界がぼけません
    mask_up = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)    
    overlay = np_img.copy()
    overlay[mask_up == 255] = [0, 0, 255]  # ここが上書き色（BGRで青）
    # ブレンド合成
    blended = cv2.addWeighted(np_img, 0.6, overlay, 0.4, 0)
    # 8. JPEGエンコードして保存（StreamingResponseの代わりに保存）
    _, jpeg = cv2.imencode(".jpg", blended)
    with open("result.jpg", "wb") as f:
        f.write(jpeg.tobytes())
    res = FileResponse(
        path=Path("result.jpg"),
        media_type="image/jpeg",
        filename="result.jpg",
    )
    print(res)
    return res 


@app.get("/api/test")
def get_test_img():
    file_path = Path("test.jpg")
    return FileResponse(
        path=file_path,
        media_type="image/jpeg",
        filename="test.jpg",
    )
if __name__ == "__main__":
    # uvicorn server_node:app --host 0.0.0.0 --port 8080
    uvicorn.run(app, host="0.0.0.0", port=8000)

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
    # ① アップロード画像を読み込む
    img = Image.open(file.file).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    # ② モデルで推論
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
        preds = output.argmax(0).byte().cpu().numpy()

    # ③ クラスID=15 のマスク作成
    mask = (preds == 15).astype(np.uint8) * 255
    np_img = np.array(img)
    overlay = np_img.copy()
    overlay[mask == 255] = [0, 0, 255]  # ここが上書き色（BGRで青）

    # ④ オーバーレイ合成
    blended = cv2.addWeighted(np_img, 0.6, overlay, 0.4, 0)

    # ⑤ JPEGにエンコード
    ok, jpeg = cv2.imencode(".jpg", blended)
    if not ok:
        # 念のためエンコード失敗時のエラー
        raise HTTPException(status_code=500, detail="Failed to encode image")

    # ⑥ ローカルに result.jpg として保存
    result_path = os.path.join(os.path.dirname(__file__), "result.jpg")
    with open(result_path, "wb") as f:
        f.write(jpeg.tobytes())

    # ⑦ result.jpg をレスポンスとして返す
    return FileResponse(
        path=result_path,
        media_type="image/jpeg",
        filename="result.jpg",
    )


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

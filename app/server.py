 # server_node.py
import io

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException,Header, status, Depends
from fastapi.responses import StreamingResponse
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from fastapi.responses import FileResponse
from pathlib import Path
import os
import requests
from dotenv import load_dotenv   
import json

load_dotenv()  # .envファイルから環境変数を読み込む
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

API_KEY_HEADER_NAME = "X-API-Key"
def get_api_key(x_api_key: str = Header(..., alias=API_KEY_HEADER_NAME)):
    """
    X-API-Key ヘッダを検証する dependency。
    不一致なら 401 を投げる。
    """
    API_KEY = os.getenv("API_KEY")
    if API_KEY is None:
        # サーバー側の設定ミスは 500 にしておく
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key is not configured on server",
        )
    if x_api_key != API_KEY:
        # ここで 401 を返す
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return x_api_key

@app.post("/api/detect")
async def detect(file: UploadFile = File(...), 
                 api_key: str = Depends(get_api_key)):
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
    bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    h0, w0 = np_img.shape[:2]
    # mask を元画像サイズに拡大/縮小,最近傍補間（INTER_NEAREST）を使うことでクラス境界がぼけません
    mask_up = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)    
    # 人間感知を判断
    person_pixels = int((mask_up == 255).sum())
    human_detected = person_pixels > 0
    print(f"person_pixels = {person_pixels}, human_detected = {human_detected}")
    # オーバーレイ画像作成
    overlay = np_img.copy()
    overlay[mask_up == 255] = [0, 0, 1]  # ここが上書き色（BGR）
    # ブレンド合成
    blended = cv2.addWeighted(np_img, 0.6, overlay, 0.4, 1)
    # 8. JPEGエンコードして保存（StreamingResponseの代わりに保存）
    _, jpeg = cv2.imencode(".jpg", bgr)
    with open("result.jpg", "wb") as f:
        f.write(jpeg.tobytes())
    
    # Discord通知
    if human_detected:
        print('人を検知しました。Discordに通知を送信します。')
        send_discord_notification(
            message="人を検知しました（GPUサーバーから）",
            image_path="result.jpg")
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

def send_discord_notification(message: str, image_path: str | Path | None = None):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        raise ValueError("DISCORD_WEBHOOK_URL is not set in environment variables.")
    # 画像付きの場合
    if image_path is not None:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        payload = {
            "content": message
        }
        with image_path.open("rb") as f:
            files = {
                # "file" という名前でファイルを送る
                "file": (image_path.name, f, "image/jpeg"),
            }
            # payload_json に JSON 文字列として content を入れる
            response = requests.post(
                webhook_url,
                data={"payload_json": json.dumps(payload)},
                files=files,
            )
    else:
        # テキストのみの場合（今まで通り）
        payload = {"content": message}
        response = requests.post(webhook_url, json=payload)
    # 画像付きだと 200、テキストだけだと 204 が返ることがある
    if response.status_code not in (200, 204):
        raise Exception(f"Failed to send notification: {response.status_code}, {response.text}")

    return response

if __name__ == "__main__":
    # uvicorn server_node:app --host 0.0.0.0 --port 8080
    uvicorn.run(app, host="0.0.0.0", port=8000)

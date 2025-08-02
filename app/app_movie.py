import io
import json
import os
import threading
import time
from datetime import datetime

import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse

# picamera2ライブラリをインポート
try:
    from picamera2 import Picamera2
    from picamera2.encoders import JpegEncoder
    from picamera2.outputs import FileOutput
except ImportError:
    print("="*50)
    print("エラー: picamera2ライブラリが見つかりません。")
    print("必要なライブラリをインストールしてください: pip install picamera2")
    print("="*50)
    exit()

# ストリーミング用のバッファークラス
# カメラからのフレームを保持し、新しいフレームが来たら通知する役割
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf # 最新フレームに更新して、古いフレームは破棄。
            self.condition.notify_all() # 待っているフレームを全員起こす

# FastAPIアプリケーションのインスタンスを作成
app = FastAPI()

# Picamera2のインスタンスをグローバルに作成
picam2 = Picamera2()
# ビデオ用の設定。解像度を少し下げて配信をスムーズにします。
config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(config)

# ストリーミング用の出力オブジェクトを作成
output = StreamingOutput()
# カメラのエンコーダーをJPEGに設定し、出力をoutputオブジェクトに指定
picam2.start_encoder(JpegEncoder(), FileOutput(output))


@app.get("/")
def index():
    content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Raspberry Pi - Camera Live Stream</title>
        <style>
            body {{ font-family: sans-serif; text-align: center; background-color: #f0f0f0; padding-top: 20px; }}
            img {{ border: 5px solid #333; border-radius: 10px; max-width: 90%; }}
            h1 {{ color: #444; }}
        </style>
    </head>
    <body>
        <h1>Raspberry Pi - Camera Live Stream</h1>
        <p>現在の時刻: <span id="clock">--:--:--</span></p>
        <img src="/api/camera/stream" width="640" height="480">

        <script>
        let offsetMs = 0;   // server_time - client_time の推定（ms）

        async function syncOnce() {
            const t0 = Date.now();
            const res = await fetch('/api/time', {cache:'no-store'});
            const j = await res.json();
            const t1 = Date.now();
            const rtt = t1 - t0;
            const serverAtMid = j.epoch_ms + (rtt/2); // 応答生成時刻を往復遅延の半分で前に進める
            offsetMs = serverAtMid - t1;              // サーバとクライアントの差分
        }

        function pad(n){ return n.toString().padStart(2,'0'); }

        function render() {
            const now = new Date(Date.now() + offsetMs);
            const h = pad(now.getHours());
            const m = pad(now.getMinutes());
            const s = pad(now.getSeconds());
            document.getElementById('clock').textContent = `${h}:${m}:${s}`;
        }

        // 初回同期 → 以降は1秒ごと更新、たまに再同期
        (async () => {
            await syncOnce();
            render();
            setInterval(render, 1000);   // 表示更新
            setInterval(syncOnce, 60_000); // 1分ごとに再同期（任意）
        })();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=content)

@app.get("/api/time")
def api_time():
    # サーバの現在時刻（エポックms）
    now_ms = int(time.time() * 1000)
    return {"epoch_ms": now_ms}


def generate_frames():
    """
    カメラからのフレームを連続的に生成するジェネレータ関数
    """
    while True:
        with output.condition:
            # 新しいフレームが来るまで待機
            output.condition.wait()
            frame = output.frame
        
        # HTTPストリーミング用の形式でフレームを返す
        # b'--frame\r\n' は各フレームの区切り文字
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get('/api/camera/stream')
def camera_stream():
    """
    MJPEGストリームを配信するエンドポイント
    """
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # カメラをバックグラウンドで起動
    picam2.start()
    
    # Uvicornサーバーを起動
    # この時点では、カメラは既に映像を生成し始めています
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # アプリケーション終了時にカメラを停止（通常はCTRL+Cで終了するため、ここは実行されないことが多い）
    picam2.stop()


import io
import time
import threading
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn

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
            self.frame = buf
            self.condition.notify_all()

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
    """
    ライブストリーミングを表示するHTMLページを返します。
    """
    content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Raspberry Pi - Camera Live Stream</title>
        <style>
            body { font-family: sans-serif; text-align: center; background-color: #f0f0f0; padding-top: 20px; }
            img { border: 5px solid #333; border-radius: 10px; max-width: 90%; }
            h1 { color: #444; }
        </style>
    </head>
    <body>
        <h1>Raspberry Pi - Camera Live Stream</h1>
        <img src="/api/camera/stream" width="640" height="480">
    </body>
    </html>
    """
    return HTMLResponse(content=content)


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


from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import io
import uvicorn
import time

# picameraライブラリをインポート
# picamera2の代わりに、より広く使われているpicameraを使用します。
# エラーが出る場合は、ターミナルで 'sudo apt-get install python3-picamera' または 'pip install "picamera[array]"' を実行してください。
try:
    from picamera import PiCamera
except ImportError:
    print("="*50)
    print("エラー: picameraライブラリが見つかりません。")
    print("ターミナルで 'pip install \"picamera[array]\"' を実行してください。")
    print("="*50)
    exit()


app = FastAPI()

@app.get("/api/camera/capture")
def capture_image():
    """
    カメラで画像を撮影し、その画像をストリーミングで返します。
    ブラウザで http://<ラズパイのIP>:8000/api/camera/capture にアクセスすると画像が表示されます。
    """
    # メモリ上のバッファ（BytesIOオブジェクト）を作成
    buffer = io.BytesIO()

    try:
        # 1. 'with'ステートメントでカメラオブジェクトを作成し、リソースを安全に管理
        with PiCamera() as camera:
            # 2. カメラの解像度を設定
            camera.resolution = (1920, 1080)

            # 3. カメラのウォームアップ時間
            # センサーが光レベルに慣れるのを待つことで、画質が向上します
            print("カメラをウォームアップしています...")
            time.sleep(2)

            # 4. 画像をメモリ上のバッファにJPEG形式でキャプチャ
            print("画像をキャプチャしています...")
            camera.capture(buffer, format='jpeg')
            print("キャプチャが完了しました。")

        # 5. バッファのポインタを先頭に戻す
        # これをしないと、レスポンスが空になります
        buffer.seek(0)

        # 6. StreamingResponseを使って画像を返す
        # media_typeを 'image/jpeg' に設定することで、ブラウザが画像として認識します
        return StreamingResponse(buffer, media_type="image/jpeg")

    except Exception as e:
        # picamera.exc.PiCameraError などのカメラ特有のエラーもここでキャッチされます
        print(f"エラーが発生しました: {e}")
        return {"error": f"カメラの撮影に失敗しました: {e}"}


@app.get("/api/value")
def get_value():
    """
    ユーザーが提供した既存のAPIエンドポイント
    """
    value = {"message": "Hello from backend!", "value": 42}
    return value


if __name__ == "__main__":
    # サーバーを起動
    # host="0.0.0.0" で、同じネットワーク内の他のデバイスからアクセス可能になります
    uvicorn.run(app, host="0.0.0.0", port=8000)

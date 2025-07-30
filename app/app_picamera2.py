from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import io
import uvicorn

# picamera2ライブラリをインポート
# エラーが出る場合は、上記のコマンドでインストールしてください
try:
    from picamera2 import Picamera2
except ImportError:
    print("="*50)
    print("エラー: picamera2ライブラリが見つかりません。")
    print("ターミナルで 'pip install picamera2' を実行してください。")
    print("="*50)
    exit()


app = FastAPI()

# カメラの初期化
# アプリケーションの起動時に一度だけ初期化すると効率的ですが、
# リソース管理が複雑になるため、この例ではリクエストごとに初期化します。
# これにより、コードがシンプルで安全になります。

@app.get("/api/camera/capture")
def capture_image():
    """
    カメラで画像を撮影し、その画像をストリーミングで返します。
    ブラウザで http://<ラズパイのIP>:8000/api/camera/capture にアクセスすると画像が表示されます。
    """
    try:
        # 1. カメラオブジェクトを作成
        picam2 = Picamera2()

        # 2. カメラのコンフィグレーションを設定
        # 高解像度の静止画用の設定を作成します
        config = picam2.create_still_configuration(main={"size": (1920, 1080)})
        picam2.configure(config)

        # 3. カメラを起動
        picam2.start()

        # 4. 画像をメモリ上のバッファ（BytesIOオブジェクト）にキャプチャ
        # ファイルに保存せず、メモリ内で直接扱います
        buffer = io.BytesIO()
        picam2.capture_file(buffer, format='jpeg')

        # 5. バッファのポインタを先頭に戻す
        # これをしないと、レスポンスが空になります
        buffer.seek(0)

        # 6. カメラを停止してリソースを解放
        picam2.close()

        # 7. StreamingResponseを使って画像を返す
        # media_typeを 'image/jpeg' に設定することで、ブラウザが画像として認識します
        return StreamingResponse(buffer, media_type="image/jpeg")

    except Exception as e:
        # カメラの初期化失敗など、エラーが発生した場合の処理
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
    # reload=Trueは開発中は便利ですが、カメラのようなハードウェアリソースを
    # 扱う際は予期せぬ動作をすることがあるため、安定運用時はFalseを推奨します。
    uvicorn.run(app, host="0.0.0.0", port=8000)


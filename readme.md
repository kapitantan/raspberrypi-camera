# 使い方
- pip install -r requirements.txt
- python app_movie.py
# 各ファイルの説明
- app_movie.py
    - カメラから取得した情報をpicamera2を使って映像をストリーミングしている
- app_picamera.py
    - picameraを使用した静止画の取得
    - 当初はpicamera2が重くインストールするとssh接続が切れて使用できなかったためpicameraを使おうとしていた。screenコマンドを使用することでインストールできたのでpicamera2の実装に切り替えた
- app_picamera2.py
    - picamera2を使用した静止画の取得
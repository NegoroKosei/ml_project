import cv2
import os

image_folder = 'imageData'  # 画像フォルダのパス
output_video = 'movie\\output_video.mp4'  # 出力する動画ファイル名
fps = 30

# 画像ファイル名を取得し、ソートして存在確認
images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".jpg")]
images.sort()

# 最初の画像でフレームサイズを取得
frame = cv2.imread(images[0])
height, width, layers = frame.shape

# 動画ファイルの設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# 画像をフレームとして追加
for img_path in images:
    frame = cv2.imread(img_path)
    if frame is not None:
        video.write(frame)
    else:
        print(f"画像 {img_path} の読み込みに失敗しました。")

video.release()
print("動画の作成が完了しました。")
# 連番画像の出力
import cv2
import numpy as np
import pandas as pd
import os

# テキストボックスの描画関数
# 画像, テキスト, 左下座標x, 左下座標y, テキストカラー, フォントサイズ, フォント太さ, ボックスカラー
def draw_textbox(img: np.ndarray,
                 text: str, x: int, y: int, text_color: tuple = (255, 255, 255), font_scale: float = 1, thickness: int = 2,
                 box_color: tuple = (255, 0, 0)) -> None:
    # テキストのサイズを取得, テキストのベースラインを取得(テキストの下端からベースラインまでの距離)
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # バウンディングボックスの座標を計算
    top_left = (x, y - text_height)  # ボックス左上
    bottom_right = (x + text_width, y)  # ボックス右下
    # ボックスを描画(フチなし)
    cv2.rectangle(img, top_left, bottom_right, box_color, -1)
    # テキストを描画
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

# 中心座標を求める関数
def calculate_center(x, y, w, h):
    return int((x * 2 + w) / 2.0), int((y * 2 + h) / 2.0)

# 画像の保存先フォルダ
upload_folder = 'imageData\\'
# フォントの設定
font=cv2.FONT_HERSHEY_SIMPLEX
# gt.txtの読み込み, for文での読み込みよりも早い
df = pd.read_csv('MOT17-02-DPM\\gt\\gt.txt', sep = ',', encoding='cp932', header=None)
# フレームの数だけ繰り返す
for frame in range(1, 601):
    # ファイル名の作成
    filename = f'{frame:06}'
    input_filename = 'MOT17-02-DPM\\img1\\' + filename + '.jpg'
    upload_filename = upload_folder + filename + '.jpg'

    # input画像ファイルが存在するか確認
    if os.path.exists(input_filename):
        # 画像の読み込み
        img = cv2.imread(input_filename)
        # BGR→RGBに変換(順番入れ替え,OpenCVの仕様)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # frameのデータのみを抽出
        frame_line = df[df.iloc[:, 0] == frame]
        # 番号リスト (0～11)を(0～255)のunit8型配列に変換して1列に生成
        classlist = classlist = np.linspace(0, 255, 12).astype(np.uint8).reshape(-1, 1)
        # カラーマップを使用して色を生成 (COLORMAP_JETを使用)
        colormap = cv2.COLORMAP_JET
        # カラーマップを適用して3列に変換
        color_map = cv2.applyColorMap(classlist, colormap).reshape(-1, 3)

        # データの数(行数)だけ繰り返す
        for i in range(len(frame_line)):
            # 長方形の座標を取得
            recx, recy, recw, rech = frame_line.iloc[i, 2:6]
            # クラスを取得
            reccl = frame_line.iloc[i, 7]
            # 中心座標を求める
            reccx, reccy = calculate_center(recx, recy, recw, rech)
            # 座表の範囲画像内に収める
            reclx = int(max(0, recx))
            recly = int(max(0, recy))
            recrx = int(min(img.shape[1], recw + recx))
            recry = int(min(img.shape[0], rech + recy))
            # 描画色を求める
            color = tuple(map(int, color_map[reccl]))
            # 長方形の描画
            cv2.rectangle(img, (reclx, recly), (recrx, recry), color, thickness=3)
            # ラベル(Class)の描画
            draw_textbox(img, str(reccl), reclx, recly, text_color=(0,0,0), font_scale=0.5, thickness=2, box_color=color)
            # 重心の描画
            if(0 <= reccx < img.shape[1] and 0 <= reccy < img.shape[0]):
                cv2.drawMarker(img, (reccx, reccy), (0, 255, 0), markerSize=20)

        # フレーム以前のデータのみ抽出
        frame_line = df[df.iloc[:, 0] <= frame]
        # idの最大値を取得
        max_id = frame_line.iloc[:, 1].max()
        # 物体番号ごとに線を引く
        for i in range(1, max_id + 1):
            # 物体番号でフィルタリング
            frame_line_cl = frame_line[frame_line.iloc[:, 1] == i]
            # DataFrame が空の場合はスキップ
            if frame_line_cl.empty:
                continue
            # 最後の行がフレームに存在しない場合はスキップ
            if frame_line_cl.iloc[-1, 0] != frame:
                continue
            # データが2つ以上ない場合はスキップ
            if len(frame_line_cl) < 2:
                continue
            # 色を取得
            reccl = frame_line_cl.iloc[0, 7]
            color = tuple(map(int, color_map[reccl]))
            for j in range(len(frame_line_cl) - 1):
                # 長方形の座標を取得
                recx1, recy1, recw1, rech1 = frame_line_cl.iloc[j, 2:6]
                rectx2, recty2, recw2, rech2 = frame_line_cl.iloc[j + 1, 2:6]
                # 中心座標を求める
                reccx1, reccy1 = calculate_center(recx1, recy1, recw1, rech1)
                reccx2, reccy2 = calculate_center(rectx2, recty2, recw2, rech2)
                # 線の描画
                cv2.line(img, (reccx1, reccy1), (reccx2, reccy2), color, thickness = 1)
        
        # 画像の保存
        # BGR→RGBに変換(順番入れ替え,OpenCVの仕様)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(upload_filename, img)
    else:
        print(f"{input_filename} は存在しません。")
        break
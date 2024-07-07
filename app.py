import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import cv2
from tensorflow.keras.models import load_model

# タイトル
st.title('ゼッケン識別デモアプリ')

# 説明
st.write('画像をアップロードしてください。アプリが画像中の数字を検出して予測します。')

# 画像アップロード
uploaded_file = st.file_uploader("画像ファイルを選択してください...", type=["png", "jpg", "jpeg"])

# モデルの読み込み
@st.cache(allow_output_mutation=True)
def load_trained_model():
    model = load_model('mnist_model.h5')
    return model

model = load_trained_model()

def preprocess_image(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
    return thresholded_image

def extract_and_predict_digit(image, contour, model):
    x, y, w, h = cv2.boundingRect(contour)
    digit_image = image[y:y+h, x:x+w]
    resized_digit_image = cv2.resize(digit_image, (28, 28))
    normalized_digit_image = resized_digit_image.astype('float32') / 255
    reshaped_digit_image = normalized_digit_image.reshape(1, 28, 28, 1)
    prediction = model.predict(reshaped_digit_image)
    return prediction[0]

if uploaded_file is not None:
    # 画像を読み込む
    image = Image.open(uploaded_file)
    processed_image = preprocess_image(image)
    
    # 輪郭を検出する
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 検出された数字の情報を格納するリスト
    digit_results = []
    
    for contour in contours:
        # 数字を予測
        prediction_probabilities = extract_and_predict_digit(processed_image, contour, model)
        predicted_digit = np.argmax(prediction_probabilities)
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # 検出された数字の情報をリストに追加
        digit_results.append({
            '数字': predicted_digit,
            '確率': prediction_probabilities,
            '座標': (x, y, w, h)
        })
    
    if digit_results:
        # 最も確率が高い数字を選択
        max_probability_result = max(digit_results, key=lambda x: np.max(x['確率']))
        
        # 予測結果を表示するためのImageDrawオブジェクトを作成
        draw = ImageDraw.Draw(image)
        x, y, w, h = max_probability_result['座標']
        draw.rectangle([x, y, x+w, y+h], outline="red", width=2)
        
        # 予測された数字と各数字の確率を表示
        st.write(f'予測された数字: {max_probability_result["数字"]}')
        st.write('各数字に対する確率:')
        probabilities = max_probability_result['確率']
        probabilities_df = pd.DataFrame({
            '数字': list(range(10)),
            '確率': probabilities
        })
        probabilities_df['確率'] = probabilities_df['確率'].apply(lambda x: f'{x:.2f}')
        st.table(probabilities_df)
        
        # 画像を表示
        st.image(image, caption='検出された数字を囲んだ画像。', use_column_width=True)
    else:
        st.write('画像から数字が検出されませんでした。')

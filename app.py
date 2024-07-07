import io
import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

def main():
    st.title('ゼッケン識別 デモ')
    file = st.file_uploader('画像をアップロード', type=['jpg', 'jpeg', 'png'])

    if file is not None:
        img = Image.open(file)
        st.image(img, caption="アップロード画像")

        # 予測
        gray_image = img.convert('L')
        resized_image = gray_image.resize((28, 28))
        image_array = np.array(resized_image).reshape(1, 28, 28, 1).astype('float32') / 255

        model = load_model('mnist_model.h5')
        prediction = model.predict(image_array)
        result = np.argmax(prediction)
        probabilities = prediction[0]

        st.write('予測値は', result, 'です。')

        st.write('各数字に対する確率:')
        p_rounded = [f'{prob:.3f}' for prob in probabilities]

        df = pd.DataFrame({
            '数字': list(range(10)),
            '確率': p_rounded
        })
        st.table(df)

if __name__ == '__main__':
    main()
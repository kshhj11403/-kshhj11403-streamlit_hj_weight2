# 남자 모델만 출력 

import streamlit as st
import numpy as np
import joblib

st.title("신체 정보를 이용한 허리둘레 예측 머신러닝 모델")
st.write("신체 정보를 입력하면 허리둘레를 예측합니다.")

model = joblib.load("weight_model_male.pkl")

st.sidebar.header("머신러닝 모델 설계 실습 (다중회귀)")

height = st.slider("키 (cm)", 140.0, 190.0, 170.0)
waist = st.slider("몸무게(kg)", 50.0, 120.0, 80.0)
hip = st.slider("엉덩이 둘레 (cm)", 85.0, 120.0, 100.0)

X = np.array([[height, waist, hip]])

if st.button("허리둘레 예측하기"):
    prediction = model.predict(X)
    st.success(f"예측 허리둘레 : {prediction[0]:.1f} cm")
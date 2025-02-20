#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/2/12 21:45
# @Author  : Ken
# @Software: PyCharm
import joblib
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd


def load_and_process_data(sheet_name, input_x):
    train = np.load(f'train_{sheet_name}.npy')
    test = np.load(f'test_{sheet_name}.npy')
    data = np.concatenate((train, test), axis=0)
    data = pd.DataFrame(data)
    x = np.array(input_x)
    print(x)
    X = data.iloc[:, 1:].values

    x_binary = x[:, -3:]
    x_binary_encoded = pd.get_dummies(pd.DataFrame(x_binary)).values

    X_continuous = X[:, :-3]
    x_continuous = x[:, :-3]
    scaler = StandardScaler()
    scaler.fit(X_continuous)
    x_continuous_scaled = scaler.transform(x_continuous)
    x_processed = np.hstack([x_continuous_scaled, x_binary_encoded])

    return x_processed

st.set_page_config(
    layout='wide'
)

# 标题,居中
st.markdown(
    "<h2 style='text-align: center;'> <span style='color: green;'>Prediction System for</span><span style='color: red;'> Chronic Kidney Disease（CKD）Stage</span></h2>",
    unsafe_allow_html=True)

st.markdown(
    "<h2 style='text-align: center; color: green;'>In Obstructive Sleep Apnea（OSA） Patients</h2>",
    unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: green;'></h1>", unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    A = st.selectbox('OSA Severity', ['Mild', 'Moderate', 'Severe'])
    if A == 'Mild':
        A = 1
    elif A == 'Moderate':
        A = 2
    else:
        A = 3
    Q = st.number_input('Apnea-Hypopnea Index(events/hour)', max_value=999.0, min_value=0.0)
    O = st.number_input('Slowest Pulse Rate(bpm)', max_value=999, min_value=0)
    E = st.number_input('Age(year)', max_value=200, min_value=0)
    U = st.number_input('Body Mass Index(kg/m²)', max_value=999.00, min_value=0.00)
    P = st.selectbox('Weight Status', ['Underweight or Healthy Weight', 'Overweight', 'Obesity'])
    if P == 'Underweight or Healthy Weight':
        P = 0
    elif P == 'Overweight':
        P = 1
    elif P == 'Obesity':
        P = 2
    B = st.selectbox('Hypertension',  ['No', 'Yes'])
    if B == 'No':
        B = 0
    else:
        B = 1
    D = st.selectbox('Diabetes Mellitus',  ['No', 'Yes'])
    if D == 'No':
        D = 0
    else:
        D = 1
    C = st.selectbox('Coronary Heart Disease',  ['No', 'Yes'])
    if C == 'No':
        C = 0
    else:
        C = 1
    F = st.number_input('Hemoglobin(g/L)', max_value=999.00, min_value=0.00)
    G = st.number_input('Lymphocyte(×10⁹/L)', max_value=999.00, min_value=0.00)
    H = st.number_input('Platelet Count(×10⁹/L)', max_value=999.00, min_value=0.00)
    R = st.number_input('Serum Albumin(g/L)', max_value=999.00, min_value=0.00)
with right:
    S = st.number_input('Alkaline Phosphatase(U/L)', max_value=999.00, min_value=0.00)
    T = st.number_input('Parathyroid Hormone(pmol/L)', max_value=999.00, min_value=0.00)
    V = st.number_input('Cystatin C(mg/L)', max_value=999.00, min_value=0.00)
    W = st.number_input('Serum Creatinine(umol/L)', max_value=999.00, min_value=0.00)
    X = st.number_input('Blood Urea Nitrogen(mmol/L)', max_value=999.00, min_value=0.00)
    Y = st.number_input('β2-Microglobulin(mg/L)', max_value=999.00, min_value=0.00)
    M = st.number_input('Total Cholesterol(mmol/L)', max_value=999.00, min_value=0.00)
    N = st.number_input('Low-Density Lipoprotein Cholesterol(mmol/L)', max_value=999.00, min_value=0.00)
    I = st.number_input('Serum Potassium(mmol/L)', max_value=999.00, min_value=0.00)
    J = st.number_input('Serum Phosphorus(mmol/L)', max_value=999.00, min_value=0.00)
    K = st.number_input('Calcium-Phosphorus Product(mmol²/L²)', max_value=999.00, min_value=0.00)
    L = st.number_input('Serum Magnesium(mmol/L)', max_value=999.00, min_value=0.00)

st.markdown("<p style='text-align: center; color: green;'></p>", unsafe_allow_html=True)

if st.button('CKD Stage Predict Outcomes'):
    ckd_model = joblib.load('CKD_best_xgb_model.pkl')
    input_x = [[A, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, B, C, D]]
    print(input_x)
    x = load_and_process_data('CKD', input_x)
    y_pred = ckd_model.predict(x)
    y_prob = ckd_model.predict_proba(x)
    print(y_pred)
    print(y_prob)

    # 合并 Stage 1 和 Stage 2 为 Stage 1-2，Stage 3、Stage 4 和 Stage 5 为 Stage 3-5
    prob_stage_1_2 = y_prob[0][0] + y_prob[0][1]  # Stage 1 和 Stage 2 的概率相加
    prob_stage_3_5 = y_prob[0][2] + y_prob[0][3] + y_prob[0][4]  # Stage 3、Stage 4 和 Stage 5 的概率相加

    # 创建新的数据框
    df = pd.DataFrame({
        'Stage': ['Stage 1-2', 'Stage 3-5'],
        'Probability': [prob_stage_1_2, prob_stage_3_5]
    })

    # 格式化概率为百分比
    df['Probability'] = df['Probability'].apply(lambda xx: str(round(xx * 100, 2)) + '%')

    # 找到最大概率所在的行索引
    max_row_idx = df['Probability'].apply(lambda xx: float(xx.replace('%', ''))).idxmax()

    # 自定义样式函数
    def highlight_max_row(row):
        return ['background-color: #FF9999' if row.name == max_row_idx else '' for _ in row]

    # 应用样式
    styled_df = df.style.apply(highlight_max_row, axis=1)

    # 在 Streamlit 中显示
    st.table(styled_df)
if st.button('Notice'):
    st.write('**This system is a model validation and testing system. It is not for clinical application without permission.**')

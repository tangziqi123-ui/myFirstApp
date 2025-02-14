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
    print(x[0])

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
    "<h2 style='text-align: center;'> <span style='color: green;'>Prediction System for </span><span style='color: red;'>Obstructive Sleep Apnea（OSA）Severity</span></h2",
    unsafe_allow_html=True)
st.markdown(
    "<h2 style='text-align: center; color: green;'>In Chronic Kidney Disease（CKD） Patients</h2",
    unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: green;'></h1>", unsafe_allow_html=True)

left, right = st.columns(2)
with left:
    B = st.selectbox('CKD Stage', ['Stage 1', 'Stage  2', 'Stage 3', 'Stage 4', 'Stage 5'])
    B = int(B.replace('Stage ', ''))
    C = st.number_input('Body Mass Index（kg/m²）', max_value=9999.00, min_value=0.00)
    A = st.selectbox('Body Type', ['Normal or Thinness', 'Overweightness', 'Obesity'])
    if A == 'Normal or Thinness':
        A = 0
    elif A == 'Overweightness':
        A = 1
    elif A == 'Obesity':
        A = 2
    K = st.selectbox('Hypertension', ['No', 'Yes'])
    if K == 'No':
        K = 0
    else:
        K = 1
    L = st.selectbox('Diabetes Mellitus', ['No', 'Yes'])
    if L == 'No':
        L = 0
    else:
        L = 1
    J = st.selectbox('Alcohol Consumption History', ['No', 'Yes'])
    if J == 'No':
        J = 0
    else:
        J = 1

with right:
    H = st.number_input('White Blood Cell(×10⁹/L)', max_value=9.5, min_value=3.5)
    I = st.number_input('Neutrophils(×10⁹/L)', max_value=6.3, min_value=1.8)
    G = st.number_input('Blood Creatinine(umol/L)', max_value=81, min_value=41)
    D = st.number_input('Cystatin C(mg/L)', max_value=1.03, min_value=0.00, )
    E = st.number_input('Glomerular Filtration Rate（mL/min/1.73m²）', max_value=9999.00, min_value=90.00)
    F = st.number_input('Total Carbon Dioxide in Blood（mmol/L）', max_value=30, min_value=20)

st.markdown("<p style='text-align: center; color: green;'></p>", unsafe_allow_html=True)

if st.button('OSA Severity Predict Outcomes'):
    osa_model = joblib.load('OSA_best_xgb_model.pkl')
    input_x = [[A, B, C, D, E, F, G, H, I, J, K, L]]
    print(input_x)
    x = load_and_process_data('OSA', input_x)
    print(x)
    y_pred = osa_model.predict(x)
    y_prob = osa_model.predict_proba(x)
    print(y_pred)
    print(y_prob)
    res_value = 'Severity: ' + str(y_pred[0])
    res_proba = 'Probability: ' + str(y_prob[0])

    df = pd.DataFrame(y_prob[0]).reset_index()
    df.columns = ['Severity', 'Probability']
    map_result = {0: 'Mild', 1: 'Moderate', 2: 'Severe'}
    df['Severity'] = df['Severity'].apply(lambda xx: map_result[xx])
    df['Probability'] = df['Probability'].apply(lambda xx: str(round(xx * 100, 2)) + '%')
    df.index += 1

    # 选择要高亮的列
    target_column = "Probability"  # 这里指定列名
    # 找到最大值所在的行索引
    max_row_idx = df[target_column].apply(lambda xx: float(xx.replace('%',''))).idxmax()
    # 自定义样式函数
    def highlight_max_row(row):
        return ['background-color: #FF9999' if row.name == max_row_idx else '' for _ in row]
    # 应用样式
    styled_df = df.style.apply(highlight_max_row, axis=1)

    # 在 Streamlit 中显示
    st.table(styled_df)
if st.button('About'):
    st.write('This system is a model validation and testing system. It is not for clinical application without permission.')

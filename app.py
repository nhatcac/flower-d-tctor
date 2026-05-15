import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
import io

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="AI Flower Midnight", page_icon="🌸", layout="centered")

# --- 2. GIAO DIỆN DARK MODE SIÊU TƯƠNG PHẢN ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117 !important;
    }
    h1, h2, h3, p, label, .stMarkdown {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #c9d1d9 !important;
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background: #ffb7c5 !important;
        color: #000000 !important;
        font-weight: 800;
        border: none;
        padding: 12px;
        box-shadow: 0 0 15px rgba(255, 183, 197, 0.4);
    }
    .result-card {
        padding: 25px;
        border-radius: 20px;
        background-color: #1c2128 !important;
        border: 2px solid #ffb7c5;
        margin-top: 25px;
    }
    .result-card h2 {
        color: #ffb7c5 !important;
    }
    .result-card p, .result-card b, .result-card span {
        color: #ffffff !important;
    }
    .stProgress > div > div > div > div {
        background-color: #ffb7c5 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HUẤN LUYỆN MÔ HÌNH ---
@st.cache_resource
def train_model():
    data = """chieu_dai_canh_hoa,chieu_rong_canh_hoa,chieu_dai_la_dai,chieu_rong_la_dai,mau_sac,do_thom,label
    5.1,2.0,6.0,3.2,1,9,0
    4.9,1.8,5.8,3.1,1,8,0
    5.3,2.1,6.2,3.4,3,9,0
    2.2,0.8,4.4,2.5,2,2,1
    2.4,0.9,4.6,2.7,2,3,1
    2.1,0.7,4.3,2.4,3,1,1
    5.8,1.7,6.4,3.0,4,6,2
    6.0,1.8,6.6,3.1,4,7,2
    5.9,1.6,6.5,2.9,3,6,2"""
    df = pd.read_csv(io.StringIO(data))
    X = df.drop('label', axis=1)
    y = df['label']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = train_model()

# --- 4. GIAO DIỆN APP ---
st.title("🌸 MIDNIGHT FLOWER AI")

st.sidebar.header("📋 THÔNG SỐ ĐẦU VÀO")
cd_canh = st.sidebar.number_input("Dài cánh hoa (cm)", 0.0, 10.0, 5.0)
cr_canh = st.sidebar.number_input("Rộng cánh hoa (cm)", 0.0, 10.0, 2.0)
cd_la = st.sidebar.number_input("Dài lá đài (cm)", 0.0, 10.0, 6.0)
cr_la = st.sidebar.number_input("Rộng lá đài (cm)", 0.0, 10.0, 3.0)

mau_dict = {1: "Màu Đỏ", 2: "Màu Vàng", 3: "Màu Trắng", 4: "Màu Tím"}
mau = st.sidebar.selectbox("Màu sắc chủ đạo", options=[1, 2, 3, 4], format_func=lambda x: mau_dict[x])
thom = st.sidebar.slider("Độ tỏa hương (1-10)", 1, 10, 5)

if st.sidebar.button("BẮT ĐẦU PHÂN TÍCH"):
    input_data = [[cd_canh, cr_canh, cd_la, cr_la, mau, thom]]
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    
    scores = model.decision_function(input_scaled)[0]
    exp_scores = np.exp(scores - np.max(scores))
    confidence = (exp_scores[prediction] / exp_scores.sum()) * 100

    flowers = {
        0: {"name": "HOA HỒNG 🌹", "trivia": "Hoa hồng đã tồn tại hơn 35 triệu năm.", "gift": "Tình yêu mãnh liệt, kỷ niệm ngày cưới."},
        1: {"name": "HOA CÚC 🌼", "trivia": "Biểu tượng của sự trường thọ và thanh cao.", "gift": "Sự hiếu thảo, tặng người thân, ông bà."},
        2: {"name": "HOA LAN 🌸", "trivia": "Có hơn 28.000 loài lan trên thế giới.", "gift": "Sự sang trọng, tặng đối tác, sếp."}
    }
    res = flowers[prediction]

    st.markdown(f"""
        <div class="result-card">
            <h2>Dự đoán: {res['name']}</h2>
            <p><b>Độ tin cậy của AI:</b> <span>{confidence:.2f}%</span></p>
            <hr style="border: 0.5px solid #30363d; margin: 15px 0;">
            <p><b>💡 Thông tin thú vị:</b> <span>{res['trivia']}</span></p>
            <p><b>🎁 Gợi ý quà tặng:</b> <span>{res['gift']}</span></p>
        </div>
    """, unsafe_allow_html=True)
    st.progress(int(confidence))

st.markdown("---")
st.caption("AI Flower Project - Developed by Perceptron Neural Network")
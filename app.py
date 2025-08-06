import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import os
import pandas as pd
from datetime import datetime

# Load model dan tools
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")
le = joblib.load("encoder.joblib")

# Atur halaman dan style CSS
st.set_page_config(page_title="Klasifikasi Tembakau", page_icon="🌿", layout="wide")

st.markdown("""
    <style>
        body {
            background-color: #f0fdf4;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
        }
        .title {
            text-align: center;
            color: #2e7d32;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Navigasi Sidebar
page = st.sidebar.radio("📚 Navigasi", ["📊 Prediksi", "ℹ️ Tentang Model", "🕑 Riwayat"])

if page == "📊 Prediksi":
    st.markdown("""
        <h1 class='title'>🌿 Klasifikasi Kualitas Tembakau</h1>
        <p style='text-align: center;'>Upload satu atau beberapa gambar daun tembakau untuk mengetahui kualitasnya.</p>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Pilih gambar daun tembakau", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    results = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert('RGB')

            with st.spinner(f'🔍 Menganalisis: {uploaded_file.name}'):
                # Preprocessing
                img = np.array(image)
                if img.shape[0] < 50 or img.shape[1] < 50:
                    st.error(f"❌ Gambar {uploaded_file.name} terlalu kecil.")
                    continue

                img = cv2.resize(img, (64, 64))
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_flat = img_gray.flatten().reshape(1, -1)
                img_scaled = scaler.transform(img_flat)

                # Prediksi
                pred = model.predict(img_scaled)[0]
                label = le.inverse_transform([pred])[0]
                results.append((uploaded_file.name, label, image))

        # Tampilkan hasil
        st.markdown("""
            <h3 style='color: #2e7d32;'>🧾 Hasil Prediksi:</h3>
        """, unsafe_allow_html=True)

        for fname, label, img in results:
            st.image(img, caption=f"📄 {fname} - Hasil: {label.upper()}", width=300)
            st.success(f"✔️ {fname}: {label.upper()}")

        # Simpan log
        history = pd.DataFrame([(f, l, datetime.now()) for f, l, _ in results], columns=["File", "Prediksi", "Waktu"])
        if not os.path.exists("riwayat_prediksi.csv"):
            history.to_csv("riwayat_prediksi.csv", index=False)
        else:
            history.to_csv("riwayat_prediksi.csv", mode='a', index=False, header=False)

        # Download hasil
        hasil_text = "\n".join([f"{f}: {l.upper()}" for f, l, _ in results])
        st.download_button(
            label="📥 Unduh Semua Hasil",
            data=hasil_text,
            file_name="hasil_klasifikasi_tembakau.txt",
            mime="text/plain"
        )

elif page == "ℹ️ Tentang Model":
    st.markdown("""
        <h2 style='color:#2e7d32;'>ℹ️ Tentang Sistem Klasifikasi</h2>
        <p>Web ini menggunakan model <strong>Support Vector Machine (SVM)</strong> untuk mengklasifikasikan gambar daun tembakau ke dalam tiga kategori kualitas: <strong>Rendah</strong>, <strong>Sedang</strong>, dan <strong>Tinggi</strong>.</p>
        <ul>
            <li><strong>Resolusi Gambar:</strong> 64x64 piksel (otomatis disesuaikan)</li>
            <li><strong>Ekstraksi Fitur:</strong> Flatten grayscale</li>
            <li><strong>Skalasi:</strong> StandardScaler</li>
            <li><strong>Model:</strong> SVC (kernel='rbf', C=10)</li>
            <li><strong>Akurasi Uji:</strong> ~79%</li>
        </ul>
        <p>Sistem ini dikembangkan dalam rangka <strong>Tugas Akhir Mahasiswa Universitas Islam Madura</strong> sebagai bentuk penerapan teknologi klasifikasi berbasis citra pada sektor pertanian tembakau.</p>
    """, unsafe_allow_html=True)

elif page == "🕑 Riwayat":
    st.markdown("<h2 style='color:#2e7d32;'>🕑 Riwayat Prediksi</h2>", unsafe_allow_html=True)
    if os.path.exists("riwayat_prediksi.csv"):
        df_log = pd.read_csv("riwayat_prediksi.csv")
        st.dataframe(df_log)
    else:
        st.info("Belum ada riwayat prediksi yang tersimpan.")

# Footer
st.markdown("""
    <hr style='border:1px solid #444;'>
    <p style='text-align:center; font-size: 14px;'>Dibuat oleh <b>TA Mahasiswa UIM</b> | Streamlit + SVM | 2025</p>
""", unsafe_allow_html=True)

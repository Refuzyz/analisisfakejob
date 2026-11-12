import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import easyocr
from pypdf import PdfReader
from PIL import Image
import re

# --- Simulasi Data & Model agar Aplikasi Mandiri ---
data = {
    'teks_lowongan': [
        "Dibutuhkan segera IT Support. Syarat: lulusan SMK TKJ, paham routing Mikrotik, penempatan Jakarta Timur.",
        "KERJA MUDAH GAJI JUTAAN! Cukup balas chat dari rumah. Transfer biaya pendaftaran member 250rb ke rekening admin.",
        "Lowongan Network Engineer. Menguasai Cisco Packet Tracer, instalasi LAN, dan konfigurasi server.",
        "Dicari kandidat pegawai BUMN tanpa tes! Kuota terbatas. Bayar biaya akomodasi tiket pesawat 1 juta, nanti diganti.",
        "Dibutuhkan staf magang teknisi komputer, terbiasa merakit PC dan instalasi Windows. Kirim CV ke email resmi.",
        "Kerja sampingan ketik captcha dibayar dollar. Syarat bayar deposit awal $50."
    ],
    'label': [0, 1, 0, 1, 0, 1]
}
df_temp = pd.DataFrame(data)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(df_temp['teks_lowongan'])
model = MultinomialNB()
model.fit(X_train_vec, df_temp['label'])
# --------------------------------------------------

# Inisialisasi EasyOCR Reader
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['id', 'en'])

reader = load_ocr()

# Fungsi Pemindai Heuristik untuk analisis mendalam teks secara keseluruhan
def analisis_heuristik(teks):
    teks_lower = teks.lower()
    indikator = []
    
    # 1. Pindai indikasi penipuan finansial
    kata_kunci_finansial = [
        "biaya pendaftaran", "biaya akomodasi", "transfer biaya", "deposit", 
        "biaya seragam", "uang jaminan", "biaya admin", "membayar", "ganti tiket"
    ]
    for kata in kata_kunci_finansial:
        if kata in teks_lower:
            indikator.append(f"Meminta sejumlah uang/dana dengan embel-embel '{kata}'")
            
    # 2. Pindai kontak tidak profesional (email gratisan untuk perusahaan formal)
    if re.search(r'\b[a-zA-Z0-9._%+-]+@(gmail|yahoo|outlook|hotmail|ymail)\.com\b', teks_lower):
         if any(x in teks_lower for x in ["bumn", "pertamina", "pt ", "bank", "chevron", "google", "shopee", "tokopedia"]):
             indikator.append("Menggunakan email tidak resmi (Gmail/Yahoo/dll) untuk rekrutmen perusahaan besar")
             
    # 3. Pindai iming-iming tidak realistis / mencurigakan
    kata_kunci_mencurigakan = [
        "tanpa tes", "pasti diterima", "gaji juta", "kerja mudah", 
        "ketik captcha", "balas chat saja", "tanpa seleksi", "pasti lulus"
    ]
    for kata in kata_kunci_mencurigakan:
        if kata in teks_lower:
            indikator.append(f"Iming-iming kemudahan tidak realistis: '{kata}'")
            
    return indikator

st.set_page_config(page_title="Pendeteksi Loker Palsu", page_icon="🔎", layout="centered")

st.title("Pendeteksi Loker Palsu (Anti-Scam) 🔎")
st.write("Sistem Machine Learning dan Analisis Heuristik Mendalam untuk mendeteksi indikasi penipuan pada berkas lowongan kerja.")

# Input manual
teks_input = st.text_area("Teks Lowongan (Manual)", placeholder="Paste teks lowongan di sini...", height=150)

# Unggah foto atau dokumen
file_input = st.file_uploader("Unggah Screenshot atau PDF Lowongan (Akan Dibaca Otomatis)", type=["png", "jpg", "jpeg", "pdf"])

# Proses pembacaan file secara real-time
teks_dari_file = ""
if file_input is not None:
    file_type = file_input.name.split('.')[-1].lower()
    
    with st.spinner("Membaca isi file..."):
        if file_type in ["png", "jpg", "jpeg"]:
            try:
                image = Image.open(file_input)
                image_np = np.array(image)
                # Lakukan OCR membaca keseluruhan gambar
                hasil_ocr = reader.readtext(image_np, detail=0)
                teks_dari_file = " ".join(hasil_ocr)
                st.info("Teks berhasil diekstrak dari gambar!")
            except Exception as e:
                st.error(f"Gagal membaca gambar: {e}")
                
        elif file_type == "pdf":
            try:
                reader_pdf = PdfReader(file_input)
                teks_list = []
                for page in reader_pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        teks_list.append(extracted)
                teks_dari_file = " ".join(teks_list)
                st.info("Teks berhasil diekstrak dari PDF!")
            except Exception as e:
                st.error(f"Gagal membaca PDF: {e}")

    # Tampilkan teks hasil ekstrak
    if teks_dari_file.strip():
        with st.expander("Lihat Hasil Teks Dokumen/Gambar Yang Dibaca"):
            st.write(teks_dari_file)

if st.button("Mulai Analisis AI"):
    teks_gabungan = (teks_input.strip() + " " + teks_dari_file.strip()).strip()

    if not teks_gabungan:
        st.warning("Silakan masukkan teks atau unggah file terlebih dahulu untuk dianalisis.")
    else:
        with st.spinner("Sedang menganalisis teks..."):
            # 1. Pindai dengan Heuristik / Rule-Based (Sangat Efektif untuk mendeteksi penipuan langsung)
            indikasi_scam = analisis_heuristik(teks_gabungan)
            
            # 2. Prediksi dengan Machine Learning
            teks_vec = vectorizer.transform([teks_gabungan])
            hasil_prediksi_ml = model.predict(teks_vec)[0]
            probabilitas = model.predict_proba(teks_vec)[0]
            prob_palsu = probabilitas[1] * 100
            prob_asli = probabilitas[0] * 100

            st.subheader("Hasil Analisis AI & Sistem Keamanan")

            # Jika salah satu sistem (ML atau Heuristik) mendeteksi bahaya
            if len(indikasi_scam) > 0 or hasil_prediksi_ml == 1:
                st.error(f"⚠️ HATI-HATI! Loker ini sangat terindikasi PALSU / PENIPUAN.")
                
                if len(indikasi_scam) > 0:
                    st.write("### Temuan Bahaya pada Dokumen/Gambar:")
                    for ind in indikasi_scam:
                        st.write(f"- 🚫 **{ind}**")
                else:
                    st.write(f"**Probabilitas Palsu (ML):** {prob_palsu:.2f}% (Pola teks menyerupasi arsip penipuan)")
                
                st.write("\n**Rekomendasi:** Jangan mengirimkan uang sepeser pun, jangan memberikan data sensitif (KTP/KK), dan lakukan kroscek ke situs/akun resmi perusahaan terkait.")
            else:
                st.success(f"✅ AMAN. Loker ini terindikasi ASLI.")
                st.write(f"**Tingkat Keyakinan Keaslian:** {prob_asli:.2f}%")
                st.write("**Alasan Analisis:** Tidak ditemukan kata kunci sensitif yang mengarah pada penipuan, serta struktur teks selaras dengan lowongan formal.")

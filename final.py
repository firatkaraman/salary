import streamlit as st
import pandas as pd

# Veri setini yükle
veri_seti = pd.read_csv("sonuclar.csv")

# Streamlit uygulamasını oluştur
st.title("Oyuncu Maaşları Tahmin Uygulaması")

# Kullanıcıdan oyuncu adını al
oyuncu_adi = st.text_input("Oyuncu Adı Girin:")

# Oyuncu adını kullanarak veriyi filtrele
secilen_oyuncu = veri_seti[veri_seti["Player"].str.lower() == oyuncu_adi.lower()]

# Eğer oyuncu bulunamazsa kullanıcıya bilgi ver
if secilen_oyuncu.empty:
    st.warning("Girdiğiniz oyuncu adı bulunamadı.")
else:
    # Oyuncunun bilgilerini göster
    st.subheader(f"{oyuncu_adi}'nin Maaş Bilgileri:")
    st.write(f"2019/2020 Sezonu Maaşı: ${secilen_oyuncu['Salary 19/20'].values[0]:,.2f}")
    st.write(f"Gelecek Sezon Tahmini Maaşı: ${secilen_oyuncu['y_pred'].values[0]:,.2f}")

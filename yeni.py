import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

# 1. Veri Yükleme Fonksiyonu
def load_data(file_path='HouseData2.xlsx'):
    """
    Excel dosyasından veriyi yükler.
    Sütun adlarını küçük harfe çevirir.
    """
    try:
        df = pd.read_excel(file_path)
        # Sütun isimlerini tümünü küçük harfe çeviriyoruz
        df.columns = df.columns.str.lower()
        # Kategorik sütunlardaki değerleri de küçük harfe çeviriyoruz (varsa)
        for col in ['ilçe', 'mahalle', 'odasayısı']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower()
        st.write("Veri başarıyla yüklendi, ilk 5 satır:")
        st.write(df.head())
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {e}")
        df = None
    return df
# Veriyi temizle ve sütun adlarını değiştir
df.rename(columns={
    'ilce': 'district',
    'mahalle': 'neighborhood',
    'fiyat': 'price',
    'tip': 'type',
    'Net Metrekare': 'net_area',
    'Bina yas': 'building_age',
    'Bina Kat': 'building_floor',
    'Esya': 'furnished',
    'Banyo Sayi': 'bathroom_count',
    'Oda Sayi': 'room_count',
    'Daire Kat': 'floor_number',
    'Isitma': 'heating',
    'Site': 'in_complex',
    'Balkon': 'has_balcony',
    'Balkon Sayisi': 'balcony_count'
}, inplace=True)

# Fiyat sütununu sayıya çevir
df['price'] = df['price'].str.replace('TL', '', regex=False).str.replace(',', '', regex=False).astype(float)

# Eksik verileri temizle
df.dropna(inplace=True)

# 2. Veri Ön İşleme Fonksiyonu
def preprocess_data(df):
    """
    - 'fiyat' sütununa göre aykırı değerleri filtreler (5. ve 95. percentil arasında)
    - 'fiyat' sütununu sayısal değere çevirir, hatalı (sayısal olmayan) değerlerden dolayı oluşabilecek sorunları giderir.
    - 'balkon' bilgisi mevcutsa, eksik değer içeren satırları kaldırır.
    - Kategorik verileri (ör. ilçe, mahalle, odasayısı) one-hot encoding ile dönüştürür.
    """
    # 'fiyat' sütununun kontrolü ve sayısal değere çevrilmesi
    if 'fiyat' not in df.columns:
        st.error("Veride 'fiyat' sütunu bulunamadı!")
        return None

    # Sayısal değere çevirmeye çalışıyoruz
    df['fiyat'] = pd.to_numeric(df['fiyat'], errors='coerce')
    df = df.dropna(subset=['fiyat'])  # Sayısal olmayan veriler varsa çıkartılıyor

    lower_bound = df['fiyat'].quantile(0.05)
    upper_bound = df['fiyat'].quantile(0.95)
    df = df[(df['fiyat'] >= lower_bound) & (df['fiyat'] <= upper_bound)]

    # Eğer 'balkon' sütunu varsa, eksik verileri kaldırıyoruz
    if 'balkon' in df.columns:
        df = df.dropna(subset=['balkon'])

    # Kategorik sütunlar: 'ilçe', 'mahalle', 'odasayısı'
    categorical_cols = ['ilçe', 'mahalle', 'odasayısı']
    for col in categorical_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)
        else:
            st.warning(f"'{col}' sütunu veride bulunamadı.")
    return df

# 3. Model Eğitim ve Karşılaştırma Fonksiyonu
def train_models(df):
    """
    Üç farklı regresyon modelini eğitir:
      - Karar Ağacı Regresyonu,
      - Destek Vektör Regresyonu (SVR): GridSearchCV ile parametre araması yapılır.
      - Yapay Sinir Ağı (MLPRegressor): Gizli katmanda 40, 70, 100 nöron denemeleri yapılır.
    Model performansları R² skoru ile ölçülür.
    """
    if 'fiyat' not in df.columns:
        st.error("Fiyat sütunu eksik!")
        return None, None, None

    X = df.drop('fiyat', axis=1)
    y = df['fiyat']

    # Eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {}
    scores = {}

    # A) Karar Ağacı
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    score_dt = r2_score(y_test, y_pred_dt)
    models['Karar Ağacı'] = dt
    scores['Karar Ağacı'] = score_dt

    # B) Destek Vektör Regresyonu (SVR) - GridSearchCV ile parametre araması
    svr = SVR()
    param_grid_svr = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
    }
    grid_svr = GridSearchCV(svr, param_grid_svr, cv=5, scoring='r2')
    grid_svr.fit(X_train, y_train)
    best_svr = grid_svr.best_estimator_
    y_pred_svr = best_svr.predict(X_test)
    score_svr = r2_score(y_test, y_pred_svr)
    models['SVR'] = best_svr
    scores['SVR'] = score_svr

    # C) Yapay Sinir Ağı (MLPRegressor) - 40, 70, 100 nöronlu modeller
    best_ann_score = -np.inf
    best_ann_model = None
    for neurons in [40, 70, 100]:
        ann = MLPRegressor(hidden_layer_sizes=(neurons,), max_iter=1000, random_state=42)
        ann.fit(X_train, y_train)
        y_pred_ann = ann.predict(X_test)
        score_ann = r2_score(y_test, y_pred_ann)
        if score_ann > best_ann_score:
            best_ann_score = score_ann
            best_ann_model = ann

    models['Yapay Sinir Ağı'] = best_ann_model
    scores['Yapay Sinir Ağı'] = best_ann_score

    # Streamlit arayüzü için kullanılacak özellik isimleri
    feature_columns = X.columns
    return models, scores, feature_columns

# 4. Streamlit Arayüzü Fonksiyonu
def streamlit_app(models, scores, feature_columns):
    """
    Kullanıcıya konut özelliklerini seçtiren ve seçilen modele göre fiyat tahmini yapan Streamlit arayüzü.
    Örnek olarak; ilçe, mahalle ve odasayısı seçenekleri sunulmaktadır.
    """
    st.title("Konut Fiyat Tahmin Uygulaması")

    st.sidebar.header("Konut Özellikleri Seçimi")
    # Örnek seçim seçenekleri – bu değerleri veri kümenize göre uyarlayabilirsiniz
    ilce_options = ['kadıköy', 'beşiktaş', 'üsküdar']
    mahalle_options = ['moda', 'levent', 'maslak']
    oda_options = ['2+1', '3+1', '4+1']

    selected_ilce = st.sidebar.selectbox("İlçe Seçiniz", ilce_options)
    selected_mahalle = st.sidebar.selectbox("Mahalle Seçiniz", mahalle_options)
    selected_oda = st.sidebar.selectbox("Oda Sayısı Seçiniz", oda_options)

    # Başlangıçta tüm özelliklerin değeri 0
    input_data = {col: 0 for col in feature_columns}

    # Dummy sütun isimlerinin oluşturulması: örneğin "ilçe_kadıköy"
    ilce_col = f"ilçe_{selected_ilce}"
    mahalle_col = f"mahalle_{selected_mahalle}"
    oda_col = f"odasayısı_{selected_oda}"

    if ilce_col in input_data:
        input_data[ilce_col] = 1
    if mahalle_col in input_data:
        input_data[mahalle_col] = 1
    if oda_col in input_data:
        input_data[oda_col] = 1

    # Kullanıcının model seçimi
    model_option = st.sidebar.selectbox("Model Seçiniz", list(models.keys()))

    if st.button("Fiyatı Tahmin Et"):
        model = models[model_option]
        input_df = pd.DataFrame([input_data])
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"{model_option} modeline göre tahmini konut fiyatı: {prediction:.2f} TL")
            st.info(f"Modelin R² skoru: {scores[model_option]:.2f}")
        except Exception as e:
            st.error(f"Tahmin sırasında hata oluştu: {e}")

# 5. Ana Program
def main():
    st.sidebar.title("Ayarlar")
    st.sidebar.write("Eğitilmiş modellerin başarı oranları (R²):")
    df = load_data("HouseData2.xlsx")
    if df is None:
        return

    df = preprocess_data(df)
    if df is None:
        return

    models, scores, feature_columns = train_models(df)
    if models is None:
        return

    for model_name, score in scores.items():
        st.sidebar.write(f"{model_name}: {score:.2f}")

    streamlit_app(models, scores, feature_columns)

if __name__ == '__main__':
    main()

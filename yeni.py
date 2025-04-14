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
    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip().str.lower()  # Sütunları normalize et

        for col in ['ilçe', 'mahalle', 'odasayısı']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower()

        st.success("✅ Veri başarıyla yüklendi, ilk 5 satır:")
        st.write(df.head())

        st.info(f"Verideki sütunlar: {df.columns.tolist()}")

    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {e}")
        df = None
    return df

# 2. Veri Ön İşleme Fonksiyonu
def preprocess_data(df):
    if 'price' not in df.columns and 'fiyat' in df.columns:
        df.rename(columns={'fiyat': 'price'}, inplace=True)

    if 'price' not in df.columns:
        st.error("Veride 'price' sütunu bulunamadı!")
        return None

    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price'])

    lower_bound = df['price'].quantile(0.05)
    upper_bound = df['price'].quantile(0.95)
    df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

    if 'balkon' in df.columns:
        df = df.dropna(subset=['balkon'])

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
    if 'price' not in df.columns:
        st.error("Price sütunu eksik!")
        return None, None, None

    X = df.drop('price', axis=1)
    y = df['price']

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

    # B) SVR
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

    # C) MLPRegressor
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

    feature_columns = X.columns
    return models, scores, feature_columns

# 4. Arayüz
def streamlit_app(models, scores, feature_columns):
    st.title("🏘️ Konut Fiyat Tahmin Uygulaması")

    st.sidebar.header("📌 Konut Özellikleri")
    ilce_options = ['kadıköy', 'beşiktaş', 'üsküdar']
    mahalle_options = ['moda', 'levent', 'maslak']
    oda_options = ['2+1', '3+1', '4+1']

    selected_ilce = st.sidebar.selectbox("İlçe Seçiniz", ilce_options)
    selected_mahalle = st.sidebar.selectbox("Mahalle Seçiniz", mahalle_options)
    selected_oda = st.sidebar.selectbox("Oda Sayısı Seçiniz", oda_options)

    input_data = {col: 0 for col in feature_columns}

    ilce_col = f"ilçe_{selected_ilce}"
    mahalle_col = f"mahalle_{selected_mahalle}"
    oda_col = f"odasayısı_{selected_oda}"

    if ilce_col in input_data:
        input_data[ilce_col] = 1
    if mahalle_col in input_data:
        input_data[mahalle_col] = 1
    if oda_col in input_data:
        input_data[oda_col] = 1

    model_option = st.sidebar.selectbox("Model Seçiniz", list(models.keys()))

    if st.button("🎯 Fiyatı Tahmin Et"):
        model = models[model_option]
        input_df = pd.DataFrame([input_data])
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"{model_option} modeline göre tahmini konut fiyatı: {prediction:,.2f} TL")
            st.info(f"📈 Modelin R² skoru: {scores[model_option]:.2f}")
        except Exception as e:
            st.error(f"Tahmin sırasında hata oluştu: {e}")

# 5. Ana Fonksiyon
def main():
    st.sidebar.title("🔧 Ayarlar")
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
        st.sidebar.write(f"✅ {model_name}: {score:.2f}")

    streamlit_app(models, scores, feature_columns)

if __name__ == '__main__':
    main()

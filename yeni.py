import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

def load_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        return None

def preprocess_data(df):
    df = df.copy()

    # Hedef sütunu kontrol et
    if 'fiyat' not in df.columns:
        st.error("'fiyat' adlı hedef sütun eksik.")
        return None, None, None

    # Sayısal olmayan sütunları label encode et
    le_dict = {}
    for col in df.columns:
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            le = LabelEncoder()
            try:
                df[col] = le.fit_transform(df[col].astype(str))
                le_dict[col] = le
            except:
                st.warning(f"{col} sütunu dönüştürülemedi.")
    
    X = df.drop("fiyat", axis=1)
    y = df["fiyat"]

    return X, y, le_dict

def train_models(df):
    X, y, _ = preprocess_data(df)
    if X is None:
        return None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {}
    scores = {}

    # Karar Ağacı
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    models['Karar Ağacı'] = dt
    scores['Karar Ağacı'] = r2_score(y_test, y_pred_dt)

    # Doğrusal Regresyon
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    models['Doğrusal Regresyon'] = lr
    scores['Doğrusal Regresyon'] = r2_score(y_test, y_pred_lr)

    # Random Forest
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    models['Random Forest'] = rf
    scores['Random Forest'] = r2_score(y_test, y_pred_rf)

    return models, scores, X.columns.tolist()

def streamlit_app(models, scores, feature_columns):
    st.title("Konut Fiyat Tahmin Uygulaması")

    st.sidebar.header("Giriş Özellikleri")

    user_input = {}
    for col in feature_columns:
        user_input[col] = st.sidebar.text_input(col)

    if st.button("Fiyat Tahmini Yap"):
        try:
            input_values = [float(user_input[col]) for col in feature_columns]
            input_df = pd.DataFrame([input_values], columns=feature_columns)

            st.subheader("Tahmin Sonuçları")
            for name, model in models.items():
                prediction = model.predict(input_df)[0]
                st.write(f"{name}: {round(prediction, 2)} TL")

            st.subheader("Model Başarı Skorları (R²)")
            for name, score in scores.items():
                st.write(f"{name}: {round(score, 4)}")
        except ValueError:
            st.error("Tüm girişleri sayısal formatta girin.")

def main():
    st.title("Veri Yükleme")
    uploaded_file = st.file_uploader("Excel dosyası yükleyin", type=["xlsx", "xls"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.write("Yüklenen Veri", df.head())

        models, scores, feature_columns = train_models(df)
        if models:
            streamlit_app(models, scores, feature_columns)

if __name__ == "__main__":
    main()

# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from io import BytesIO

st.set_page_config(page_title="Energy Efficiency - Buildings", layout="wide")
st.title("🏠 Energy Efficiency Prediction (Heating & Cooling Load)")

# ---------- helper functions ----------
@st.cache_data
def load_uci_dataset():
    """Load UCI ENB2012 dataset from url (xlsx). If URL blocked, the app will show instructions."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    try:
        df = pd.read_excel(url)
    except Exception as e:
        st.error("Could not download UCI dataset automatically. Please use the file uploader (Excel/CSV) or place the dataset in data/ folder.")
        raise e
    # Rename columns (UCI sheet has no friendly headers)
    df.columns = ["Relative_Compactness","Surface_Area","Wall_Area","Roof_Area",
                  "Overall_Height","Orientation","Glazing_Area","Glazing_Area_Distribution",
                  "Heating_Load","Cooling_Load"]
    return df

def read_user_file(uploaded_file):
    if uploaded_file is None:
        return None
    fname = uploaded_file.name.lower()
    if fname.endswith(".xlsx") or fname.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    elif fname.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        st.error("Unsupported file type. Upload CSV or Excel.")
        return None

def preprocess_df(df):
    # Try to ensure required columns exist; rename if possible
    expected = {"Relative_Compactness","Surface_Area","Wall_Area","Roof_Area",
                "Overall_Height","Orientation","Glazing_Area","Glazing_Area_Distribution",
                "Heating_Load","Cooling_Load"}
    # Basic cleaning
    df = df.copy()
    df = df.dropna(how="all")  # drop empty rows
    # If dataset has similar headers but different names, user must ensure correct columns.
    found = set(df.columns)
    if not expected.issubset(found):
        # If Heating/Cooling columns missing, attempt lowercase match
        lower_map = {c.lower(): c for c in df.columns}
        mapping = {}
        for e in expected:
            if e.lower() in lower_map:
                mapping[lower_map[e.lower()]] = e
        if mapping:
            df = df.rename(columns=mapping)
    return df

# ---------- Sidebar: dataset options ----------
st.sidebar.header("Dataset options")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (Excel or CSV). If you don't upload one, the UCI dataset will be used.", type=["xlsx","xls","csv"])
merge_with_uci = st.sidebar.checkbox("If I upload a dataset, merge it with UCI dataset (append rows)", value=False)
use_local_file = st.sidebar.checkbox("Load dataset from local file path (data/user_dataset.xlsx)", value=False)
local_path = "data/user_dataset.xlsx"

# ---------- Load data logic ----------
df_user = read_user_file(uploaded_file) if uploaded_file else None
df_uci = None
df = None

# Try to load UCI dataset (if needed)
if (uploaded_file is None) or (uploaded_file is not None and merge_with_uci) or use_local_file:
    try:
        df_uci = load_uci_dataset()
    except Exception:
        df_uci = None

if use_local_file and os.path.exists(local_path):
    try:
        df_local = pd.read_excel(local_path) if local_path.endswith(("xlsx","xls")) else pd.read_csv(local_path)
        df_local = preprocess_df(df_local)
        if df_uci is None:
            df = df_local
        else:
            df = pd.concat([df_uci, df_local], ignore_index=True)
    except Exception as e:
        st.error(f"Failed to read local file at {local_path}: {e}")
else:
    if df_user is not None:
        df_user = preprocess_df(df_user)
        if merge_with_uci and df_uci is not None:
            df = pd.concat([df_uci, df_user], ignore_index=True)
        else:
            df = df_user
    else:
        df = df_uci

if df is None:
    st.stop()

# show dataset
st.subheader("📊 Dataset preview (first 10 rows)")
st.dataframe(df.head(10))

# ---------- Check for required target columns ----------
required_cols = ["Heating_Load", "Cooling_Load"]
if not all(c in df.columns for c in required_cols):
    st.error("Dataset does not contain both 'Heating_Load' and 'Cooling_Load' columns. Please upload/prepare the dataset in the same format as the UCI ENB2012 dataset.")
    st.stop()

# ---------- Features & preprocessing ----------
feature_cols = ["Relative_Compactness","Surface_Area","Wall_Area","Roof_Area",
                "Overall_Height","Orientation","Glazing_Area","Glazing_Area_Distribution"]
for c in feature_cols:
    if c not in df.columns:
        st.error(f"Missing feature column: {c}. Ensure your dataset uses the same column names as the UCI dataset.")
        st.stop()

X = df[feature_cols].astype(float)
y = df[required_cols].astype(float)

# split
test_size = st.sidebar.slider("Test set proportion (%)", 5, 50, 20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42)

# ---------- Train model ----------
st.subheader("⚙️ Train model")
n_estimators = st.sidebar.slider("n_estimators (RandomForest)", 50, 500, 200, step=10)
train_button = st.button("Train RandomForest model now")

model_path = "models/energy_model.pkl"
if train_button:
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    with st.spinner("Training model..."):
        model.fit(X_train, y_train)
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, model_path)
    st.success(f"Model trained and saved to {model_path}")

# If model exists, load it (so UI can be used without retraining)
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    # if user didn't train, train automatically (small convenience)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)

# ---------- Evaluate ----------
st.subheader("📈 Evaluation on test set")
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
st.write(f"R² score: **{r2:.4f}**")
st.write(f"RMSE (multi-output averaged): **{rmse:.4f}**")

# show sample predictions
sample_df = X_test.copy().reset_index(drop=True).head(5)
sample_preds = model.predict(sample_df)
st.write("Sample test inputs and model predictions (first 5):")
res = sample_df.copy()
res["Pred_Heating"] = sample_preds[:,0]
res["Pred_Cooling"] = sample_preds[:,1]
st.dataframe(res)

# ---------- Predict from user input ----------
st.subheader("🔮 Make a custom prediction")
# give sliders based on dataset ranges
def minmax(col):
    return float(X[col].min()), float(X[col].max()), float(X[col].median())

input_data = {}
cols = {}
for col in feature_cols:
    mn, mx, med = minmax(col)
    if col in ["Orientation","Glazing_Area_Distribution"]:
        # discrete choices
        choices = sorted(list(X[col].unique()))
        input_data[col] = st.selectbox(col, choices, index=choices.index(int(med)) if int(med) in choices else 0)
    else:
        input_data[col] = st.slider(col, min_value=mn, max_value=mx, value=med)

input_df = pd.DataFrame([input_data])[feature_cols]

if st.button("Predict for above input"):
    pred = model.predict(input_df)[0]
    st.success(f"Predicted Heating Load: **{pred[0]:.2f}**")
    st.success(f"Predicted Cooling Load: **{pred[1]:.2f}**")

# ---------- Download model ----------
st.subheader("⬇️ Download trained model")
with open(model_path, "rb") as f:
    bytes_model = f.read()
st.download_button(label="Download model (.pkl)", data=bytes_model, file_name="energy_model.pkl")

st.write("App finished. Tip: If you uploaded a custom dataset, make sure its columns match the UCI dataset column names.")

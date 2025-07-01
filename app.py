import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler

# --- App Config ---
st.set_page_config(
    page_title="🌾 Crop Recommendation System",
    page_icon="🌱",
    layout="centered"
)

# --- Custom CSS for Light Theme ---
st.markdown("""
<style>
    :root {
        --primary: #2e7d32;
        --secondary: #ffffff;  /* Pure white background */
        --text: #333333;
        --card-bg: #ffffff;
    }

    body {
        color: var(--text);
        background-color: var(--secondary);
    }

    .stSlider [data-baseweb="slider"] {
        background-color: var(--primary);
    }

    .st-expander {
        background-color: var(--card-bg);
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .prediction-card {
        background-color: var(--card-bg);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.08);
        border-left: 5px solid var(--primary);
    }

    .stButton>button {
        background-color: var(--primary);
        color: white;
        font-weight: bold;
    }

    .footer {
        color: #666666;
        font-size: 0.9rem;
        text-align: center;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model and Scaler ---
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model files not found! Please check `model.pkl` and `scaler.pkl`.")
    st.stop()

# --- Check Model Features ---
if model.n_features_in_ != 6:
    st.error(f"⚠️ Model expects {model.n_features_in_} features, but this app provides 6.")
    st.stop()

# --- Title ---
st.title("🌾 Crop Recommendation System")
st.markdown("*Get science-based crop recommendations for your agricultural land*")

# --- Input Fields ---
st.header("📊 Enter Field Parameters")
col1, col2 = st.columns(2)

with col1:
    with st.expander("🧪 Soil Composition", expanded=True):
        n = st.slider("Nitrogen (N) - ppm", 0, 120, 50)
        p = st.slider("Phosphorus (P) - ppm", 0, 120, 50)
        k = st.slider("Potassium (K) - ppm", 0, 120, 50)

with col2:
    with st.expander("🌦️ Climate Conditions", expanded=True):
        ph = st.slider("Soil pH Level", 3.0, 10.0, 6.5, step=0.1)
        temp = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, step=0.5)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0, step=1.0)

# --- Predict Button ---
if st.button("🔍 Analyze & Recommend", type="primary", use_container_width=True):
    try:
        input_data = [[n, p, k, temp, humidity, ph]]
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        confidence = model.predict_proba(scaled_input).max() * 100

        st.markdown(f"""
        <div class="prediction-card">
            <h3 style='color: var(--primary); margin-bottom: 0.5rem;'>RECOMMENDED CROP</h3>
            <h2 style='margin-top: 0; color: var(--text);'>{prediction.upper()}</h2>
            <p style='color: #555555;'>Confidence: <strong>{confidence:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div class="footer">
    <p><strong>Hajee Mohammad Danesh Science & Technology University</strong></p>
</div>
""", unsafe_allow_html=True)

import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler

# --- App Config --
st.set_page_config(
    page_title="🌾 HSTU Crop Expert",
    page_icon="🌱",
    layout="centered"
)

# Custom CSS for optimal readability
st.markdown("""
<style>
    /* Base colors */
    :root {
        --primary: #2e7d32;
        --secondary: #f5f5f5;
        --text: #333333;
        --card-bg: #ffffff;
    }
    
    /* Text contrast */
    body {
        color: var(--text);
        background-color: var(--secondary);
    }
    
    /* Slider styling */
    .stSlider [data-baseweb="slider"] {
        background-color: var(--primary);
    }
    
    /* Input containers */
    .st-expander {
        background-color: var(--card-bg);
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Prediction card */
    .prediction-card {
        background-color: var(--card-bg);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-left: 5px solid var(--primary);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: var(--primary);
        color: white;
        font-weight: bold;
    }
    
    /* Footer styling */
    .footer {
        color: #666666;
        font-size: 0.9rem;
        text-align: center;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Resources ---
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model files not found! Please check model.pkl and scaler.pkl")
    st.stop()

# Verify model expects 6 features (without rainfall)
if model.n_features_in_ != 6:
    st.error(f"⚠️ Model expects {model.n_features_in_} features, but app is designed for 6 features")
    st.stop()

# --- Header ---
st.title("🌾 HSTU Smart Crop Advisor")
st.markdown("""
*Get science-based crop recommendations for your agricultural land*
""")

# --- Input Section ---
st.header("📊 Enter Field Parameters")

col1, col2 = st.columns(2)

with col1:
    with st.expander("🧪 Soil Composition", expanded=True):
        n = st.slider("Nitrogen (N) - ppm", 0, 100, 50)
        p = st.slider("Phosphorus (P) - ppm", 0, 100, 50)
        k = st.slider("Potassium (K) - ppm", 0, 100, 50)

with col2:
    with st.expander("🌦️ Climate Conditions", expanded=True):
        ph = st.slider("Soil pH Level", 3.0, 10.0, 6.5, step=0.1)
        temp = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, step=0.5)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0, step=1.0)

# --- Prediction Section ---
if st.button("🔍 Analyze & Recommend", type="primary", use_container_width=True):
    try:
        # Create input array with 6 features (no rainfall)
        input_data = [[n, p, k, temp, humidity, ph]]
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        confidence = model.predict_proba(scaled_data).max() * 100
        
        st.markdown(f"""
        <div class="prediction-card">
            <h3 style='color: var(--primary); margin-bottom: 0.5rem;'>RECOMMENDED CROP</h3>
            <h2 style='margin-top: 0; color: var(--text);'>{prediction.upper()}</h2>
            <p style='color: #555555;'>Confidence: <strong>{confidence:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.balloons()
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")

# --- Team Copyright ---
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Developed with ❤️ by <strong>HSTU_KichuValoLageNa</strong></p>
    <p>© 2024 Hajee Mohammad Danesh Science & Technology University</p>
</div>
""", unsafe_allow_html=True)
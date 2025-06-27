import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ---Configuración Inicial de la Página ---
st.set_page_config(
    page_title="Monitor de Salud Cardiovascular",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 1. Estilos CSS Personalizados para un Diseño Moderno y Profesional ---

st.markdown("""
<style>
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333333;
        background-color: #f0f2f6;
    }
    .main .block-container {
        max-width: 1100px;
        padding: 2rem 3rem;
        margin: auto;
    }

    h1 { font-size: 2.8em; color: #CD5C5C; }
    h2 { font-size: 2.2em; color: #4682B4; }
    h3 { font-size: 1.8em; color: #5F9EA0; }

    .stButton>button {
        background-color: #28a745;
        color: white;
        font-weight: bold;
        padding: 0.75em 1.5em;
        border-radius: 0.5em;
        border: none;
        transition: background-color 0.3s ease, transform 0.2s ease;
        font-size: 1.1em;
        width: 100%;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #218838;
        transform: translateY(-2px);
    }

    

    .result-title-main {
        font-size: 2.2em;
        font-weight: bold;
        margin: 0.5em 0 1em 0;
        text-align: center;
    }

    .risk-low-color { color: #28a745; }
    .risk-high-color { color: #dc3545; }

    .recommendations-box {
        margin-top: 2em;
        padding: 25px;
        background-color: #f8f8f8;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        text-align: left;
    }
    .recommendations-box ul {
        list-style-type: disc;
        margin-left: 20px;
    }
    .recommendations-box li {
        margin-bottom: 0.5em;
        color: #333333;
    }

  
</style>
""", unsafe_allow_html=True)


# ---Cargar modelo ---

@st.cache_resource
def load_assets():
    model_loaded = None
    scaler_loaded = None

 
    try:
        with open("cardio_model.pkl", 'rb') as file:
            model_loaded = pickle.load(file)
    except FileNotFoundError:
        st.sidebar.error(f"Error: Archivo del modelo cardio_model.pkl no encontrado. Asegúrate de que esté en la carpeta correcta.")

    try:
        with open("scaler.pkl", 'rb') as file:
            scaler_loaded = pickle.load(file)
    except FileNotFoundError:
        st.sidebar.warning(f"Advertencia: Archivo del scaler scaler.pkl no encontrado. Las predicciones podrían ser inexactas sin el escalado adecuado.")
    
    return model_loaded, scaler_loaded

# Cargar los recursos al inicio de la aplicación
model, scaler = load_assets()



st.title("Calcula tu riesgo cardiovascular")
st.markdown("El riesgo cardiovascular indica las posibilidades que tienes de sufrir alguna enfermedad cardiaca según tus antecedentes y hábitos de vida.")


FEATURE_COLUMNS_ORDER = [
    'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
    'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'age_years',"imc"]

# Inicio del formulario
with st.form("cardio_prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Datos Demográficos")
        edad_years = st.slider("Edad", 18, 100, 50,)
        edad_model_input = edad_years 

        genero_display = st.radio("Sexo", options=["Femenino", "Masculino"], index=0,horizontal=True)
        genero_encoded = 0 if genero_display == "Femenino" else 1 

        altura = st.number_input("Altura (cm)", min_value=50.0, max_value=250.0, value=170.0, step=1.0,)
        peso = st.number_input("Peso (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.5)

    with col2:
        st.subheader("Mediciones Clínicas")
        presion_sistolica = st.number_input("Presión Sistólica (mmHg)", min_value=70, max_value=250, value=120, step=1, help="El primer número de tu lectura de presión arterial (sistólica).")
        presion_diastolica = st.number_input("Presión Diastólica (mmHg)", min_value=40, max_value=180, value=80, step=1, help="El segundo número de tu lectura de presión arterial (diastólica).")

        colesterol_map = {1:'Normal', 2:'Elevado', 3:'Muy Elevado'}
        colesterol_input = st.selectbox(
            "Nivel de Colesterol",
            options=list(colesterol_map.keys()),
            format_func=lambda x: colesterol_map[x],
            index=0, 
            help="Nivel de colesterol según categorías médicas (1: Normal, 2: Elevado, 3: Muy Elevado)."
            )

        glucosa_map = {1:'Normal', 2:'Elevada', 3:'Muy Elevada'}
        glucosa_input = st.selectbox(
            "Nivel de Glucosa",
            options=list(glucosa_map.keys()),
            format_func=lambda x: glucosa_map[x],
            index=0, 
            help="Nivel de glucosa en sangre (1: Normal, 2: Elevada, 3: Muy Elevada)."
            )

    with col3:
        st.subheader("Hábitos de Vida")
        fuma_input = st.radio("¿Fumas?", options=["No", "Sí"], index=0,horizontal=True)
        fuma_encoded = 0 if fuma_input == "No" else 1

        alcohol_input = st.radio("¿Consumes alcohol?", options=["No", "Sí"], index=0,horizontal=True)
        alcohol_encoded = 0 if alcohol_input == "No" else 1

        activo_input = st.radio("¿Realizas actividad física?", options=["No activo", "Activo"], index=0,horizontal=True)
        activo_encoded = 0 if activo_input == "No activo" else 1


    submitted = st.form_submit_button("Calcular")

    # --- Lógica de Predicción  ---
# --- Lógica de Predicción y Visualización ---
if submitted:
    if model is not None:
        bmi = 0.0
        if altura > 0: 
            bmi = peso / ((altura / 100)**2)
            user_data_raw = {
                'gender': genero_encoded,
                'height': altura,
                'weight': peso,
                'ap_hi': presion_sistolica,
                'ap_lo': presion_diastolica,
                'cholesterol': colesterol_input,
                'gluc': glucosa_input,
                'smoke': fuma_encoded,
                'alco': alcohol_encoded,
                'active': activo_encoded,
                'age_years': edad_model_input,
                'imc': bmi
            }

            input_df = pd.DataFrame([user_data_raw], columns=FEATURE_COLUMNS_ORDER)

            # Escalar los datos
            processed_input = scaler.transform(input_df) if scaler else input_df.values

            # Predicción
            try:
                prediccion_clase = model.predict(processed_input)[0]
                probabilidad_enfermedad = model.predict_proba(processed_input)[:, 1][0]
            except Exception as e:
                st.error(f"Error al realizar la predicción. Detalle: {e}")
                st.stop()

            # --- Mostrar resultados según rangos personalizados ---
            umbral_decision = 0.5
            umbral_moderado_inferior = 0.35
            umbral_moderado_superior = 0.65

            if probabilidad_enfermedad >= umbral_moderado_superior:
                # Riesgo Alto
                st.markdown(f""" 
                <div class="result-title-main risk-high-color"> 
                    ¡ALERTA! Tu riesgo es ALTO. 
                </div> 
                <p style="text-align: center; font-size:1.1em; color: #dc3545;"> 
                    La probabilidad estimada de riesgo es del <b>{probabilidad_enfermedad:.2%}</b>. 
                </p> 
                """, unsafe_allow_html=True)

                st.markdown(""" 
                <div class="recommendations-box"> 
                    <p>Según tus datos, tienes un <b>riesgo cardiovascular elevado</b>. Se recomienda consultar con un médico lo antes posible.</p> 
                    <p><b>Recomendaciones:</b></p> 
                    <ul> 
                        <li>Controla tu presión arterial, colesterol y glucosa.</li> 
                        <li>Haz cambios en tu dieta y estilo de vida.</li> 
                        <li>Consulta con un profesional de salud.</li> 
                    </ul> 
                </div> 
                """, unsafe_allow_html=True)

            elif probabilidad_enfermedad >= umbral_moderado_inferior:
                # Riesgo Moderado
                st.markdown(f""" 
                <div class="result-title-main" style="color:#FF8C00;"> 
                    Riesgo moderado 
                </div> 
                <p style="text-align: center; font-size:1.1em; color: #FF8C00;"> 
                    La probabilidad estimada de riesgo es del <b>{probabilidad_enfermedad:.2%}</b>. 
                </p> 
                """, unsafe_allow_html=True)

                st.markdown(""" 
                <div class="recommendations-box"> 
                    <p>Tu riesgo es moderado. Aunque no es crítico, es importante tomar medidas preventivas.</p> 
                    <p><b>Recomendaciones:</b></p> 
                    <ul> 
                        <li>Revisa tu dieta y estilo de vida.</li> 
                        <li>Haz ejercicio regularmente y mantente activo.</li> 
                        <li>Considera un chequeo médico preventivo.</li> 
                    </ul> 
                </div> 
                """, unsafe_allow_html=True)

            else:
                # Riesgo Bajo
                st.markdown(f""" 
                <div class="result-title-main risk-low-color"> 
                    ¡Tu riesgo es bajo! 
                </div> 
                <p style="text-align: center; font-size:1.1em; color: #28a745;"> 
                    La probabilidad estimada de riesgo es del <b>{probabilidad_enfermedad:.2%}</b>. 
                </p> 
                """, unsafe_allow_html=True)

                st.markdown(""" 
                <div class="recommendations-box"> 
                    <p>El modelo indica que tu riesgo cardiovascular es bajo. ¡Sigue cuidándote!</p> 
                    <p><b>Recomendaciones:</b></p> 
                    <ul> 
                        <li>Mantén hábitos saludables.</li> 
                        <li>Haz actividad física y evita el tabaco.</li> 
                        <li>Realiza controles médicos periódicos.</li> 
                    </ul> 
                </div> 
                """, unsafe_allow_html=True)

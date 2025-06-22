import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --- 0. Configuración Inicial de la Página ---
st.set_page_config(
    page_title="Monitor de Salud Cardiovascular",
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Fuente general y colores de texto */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; /* Fuente limpia y legible */
        color: #333333;
        background-color: #f0f2f6; /* Un gris claro suave para el fondo */
    }
    .main .block-container {
        padding-top: 2rem;
        padding-right: 3rem;
        padding-left: 3rem;
        padding-bottom: 2rem;
        min-height: calc(100vh - 100px); /* Ajuste para dejar espacio al footer */
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2F4F4F; /* Tono oscuro para encabezados */
        font-weight: 600;
        border-bottom: 1px solid #e0e0e0; /* Línea sutil bajo los títulos para separación */
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    h1 { font-size: 2.8em; color: #CD5C5C; /* Rojo suave, profesional para el título principal */ }
    h2 { font-size: 2.2em; color: #4682B4; /* Azul acero para secciones */ }
    h3 { font-size: 1.8em; color: #5F9EA0; /* Azul verdoso para sub-secciones */ }

    /* Botones */
    .stButton>button {
        background-color: #28a745; /* Verde estándar para acciones principales */
        color: white;
        font-weight: bold;
        padding: 0.75em 1.5em;
        border-radius: 0.5em;
        border: none;
        transition: background-color 0.3s ease, transform 0.2s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        font-size: 1.1em;
        width: 100%; /* Botón de predicción ancho para mayor impacto */
    }
    .stButton>button:hover {
        background-color: #218838;
        transform: translateY(-2px);
    }

    /* Mensajes de feedback (éxito, error, advertencia, info) */
    .stAlert {
        border-radius: 8px;
        padding: 20px;
        font-size: 1.05em; /* Un poco más pequeño para ser sutil */
        line-height: 1.6;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15); /* Sombra para dar profundidad */
        margin-bottom: 20px;
    }
    .stAlert.st-success {
        background-color: #e6ffe6;
        color: #006600;
        border-left: 6px solid #28a745; /* Borde verde fuerte */
    }
    .stAlert.st-error {
        background-color: #ffe6e6;
        color: #cc0000;
        border-left: 6px solid #dc3545; /* Borde rojo fuerte */
    }
    .stAlert.st-warning {
        background-color: #fff3cd;
        color: #856404;
        border-left: 6px solid #ffc107; /* Borde amarillo fuerte */
    }
    .stAlert.st-info {
        background-color: #e2f2ff;
        color: #004085;
        border-left: 6px solid #007bff; /* Borde azul fuerte */
    }

    /* Input widgets */
    .stSlider > div > div > div { /* Estilo para el thumb del slider */
        background-color: #4682B4 !important;
    }
    .stSlider > div > div { /* Estilo para la barra del slider */
        background-color: #ADD8E6 !important;
    }
    .stNumberInput, .stSelectbox, .stRadio {
        margin-bottom: 1rem; /* Espacio entre inputs */
    }

    /* Sidebar */
    .css-1d391kg { /* Target sidebar background */
        background-color: #F8F8F8; /* Un gris muy claro para el sidebar */
        padding-top: 2rem;
    }
    .css-1lcbmhc { /* Target sidebar text */
        color: #555555;
    }
    .css-1lcbmhc h2 {
        color: #4682B4; /* Color de título en sidebar */
    }
    .css-1lcbmhc .stRadio > label { /* Estilo para las opciones del radio en el sidebar */
        font-size: 1.05em; /* Ligeramente más grande para legibilidad */
        padding: 0.5em 0;
    }

    /* Footer - Fijo en la parte inferior de la ventana */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #333333; /* Color oscuro para el footer */
        color: white;
        text-align: center;
        padding: 15px 0;
        font-size: 0.85em; /* Tamaño de fuente ligeramente más pequeño */
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1); /* Sombra sutil arriba */
        z-index: 1000; /* Asegurar que esté por encima de otros elementos */
    }
    .footer a {
        color: #ADD8E6; /* Color de enlace en el footer */
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)


# --- 2. Carga del Modelo y Scaler (con caché para eficiencia) ---
MODEL_FILENAME = 'cardio_model.pkl'
SCALER_FILENAME = 'scaler.pkl'

@st.cache_resource
def load_assets():
    model_loaded = None
    scaler_loaded = None

    # Cargar modelo
    try:
        with open(MODEL_FILENAME, 'rb') as file:
            model_loaded = pickle.load(file)
        st.sidebar.success("Modelo predictivo cargado con éxito.")
    except FileNotFoundError:
        st.sidebar.error(f"Error: Archivo del modelo '{MODEL_FILENAME}' no encontrado. Asegúrate de que esté en la carpeta correcta.")
        st.stop() 

    # Cargar scaler
    try:
        with open(SCALER_FILENAME, 'rb') as file:
            scaler_loaded = pickle.load(file)
        st.sidebar.success("StandardScaler cargado con éxito.")
    except FileNotFoundError:
        st.sidebar.warning(f"Advertencia: Archivo del scaler '{SCALER_FILENAME}' no encontrado. Las predicciones podrían ser inexactas sin el escalado adecuado.")
    
    
    return model_loaded, scaler_loaded

model, scaler = load_assets()


# --- 2. Título Principal y Descripción General de la App ---
st.title("Monitor de Salud Cardiovascular Personalizado")
st.markdown("""
Una herramienta avanzada para estimar tu **riesgo cardiovascular** utilizando un modelo de Machine Learning.
""")
st.info("Nota Importante: Este sistema es una herramienta de apoyo y NO reemplaza el consejo, diagnóstico o tratamiento médico profesional. Siempre consulta a un especialista de la salud.")

st.markdown("---") # Separador visual

# --- 3. Sidebar como Menú de Navegación ---
st.sidebar.title("Menú Principal")
selection = st.sidebar.radio("Ir a:", ["Análisis de Riesgo", "Acerca del Modelo"])

if selection == "Análisis de Riesgo":
    st.header("Datos del Paciente")
    st.markdown("Introduce la información solicitada a continuación. La exactitud de los datos mejora la precisión del análisis.")
    FEATURE_COLUMNS_ORDER = [
        'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
        'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'age_years',"imc"
    ]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Datos Demográficos")
        edad_years = st.slider("Edad (años)", 18, 100, 50, help="Tu edad en años cumplidos.")
        edad_model_input = edad_years 

        genero_display = st.radio("Género", options=["Femenino", "Masculino"], index=0, help="Selecciona tu género biológico.")
        genero_encoded = 0 if genero_display == "Femenino" else 1 

        altura = st.number_input("Altura (cm)", min_value=50.0, max_value=250.0, value=170.0, step=1.0, help="Tu altura en centímetros.")
        peso = st.number_input("Peso (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.5, help="Tu peso actual en kilogramos.")

    with col2:
        st.subheader("Mediciones Clínicas Clave")
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
        fuma_input = st.radio("¿Fumas actualmente?", options=["No", "Sí"], index=0, help="Indica si eres fumador.")
        fuma_encoded = 0 if fuma_input == "No" else 1

        alcohol_input = st.radio("¿Consumes alcohol regularmente?", options=["No", "Sí"], index=0, help="Indica si consumes alcohol de forma habitual.")
        alcohol_encoded = 0 if alcohol_input == "No" else 1

        activo_input = st.radio("¿Realizas actividad física?", options=["No activo", "Activo"], index=1, help="Indica si eres físicamente activo (ejercicio regular).")
        activo_encoded = 0 if activo_input == "No activo" else 1

    st.markdown("---") 

    # --- 4. Botón de Predicción ---
    st.header("Obtener Análisis")
    if st.button("Analizar Mi Riesgo Cardiovascular", key="predict_button"): 
        if model is not None:
            bmi = 0.0
            if altura > 0:
                bmi = peso / ((altura / 100)**2)
            
            st.info(f"Calculando tu Índice de Masa Corporal (IMC): **{bmi:.2f}**") 
            st.markdown("---") 
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
                'age_years': edad_model_input
            }

            input_df = pd.DataFrame([user_data_raw], columns=FEATURE_COLUMNS_ORDER)

            # Escalar los datos de entrada
            processed_input = None
            if scaler is not None:
                try:
                    processed_input = scaler.transform(input_df)
                except Exception as e:
                    st.error(f"Error al escalar los datos. Asegúrate de que todas las características numéricas sean correctas. Detalle: {e}")
                    st.stop()
            else:
                processed_input = input_df.values
            try:
                prediccion_clase = model.predict(processed_input)[0]
                probabilidad_enfermedad = model.predict_proba(processed_input)[:, 1][0]
            except Exception as e:
                st.error(f"Error al realizar la predicción. Detalle: {e}")
                st.stop()

            st.subheader("Tu Resultado de Riesgo Cardiovascular")

            umbral_decision = 0.5 

            if probabilidad_enfermedad >= umbral_decision:
                st.error(f"""
                ### ¡ALERTA! Riesgo ALTO de enfermedad cardiovascular.
                <p style="font-size:1.1em;">
                La probabilidad estimada de riesgo es del <b>{probabilidad_enfermedad:.2%}</b>.
                </p>
                """)
                st.warning("""
                Basado en la información proporcionada, el modelo sugiere que tienes un **riesgo elevado**. Te recomendamos enfáticamente
                que **consultes a un profesional de la salud** (médico de cabecera o cardiólogo) para una evaluación y
                asesoramiento personalizado.
                """)
                st.markdown("""
                **Recomendaciones generales para un riesgo alto:**
                * Monitoriza tu presión arterial y niveles de colesterol/glucosa regularmente.
                * Adopta una dieta DASH (Dietary Approaches to Stop Hypertension) o mediterránea.
                * Aumenta tu actividad física con al menos 150 minutos de ejercicio moderado a la semana.
                * Abandona el hábito de fumar.
                * Modera o elimina el consumo de alcohol.
                * Gestiona el estrés de forma activa.
                """)
            else:
                st.success(f"""
                ### ¡Buenas Noticias! Bajo riesgo de enfermedad cardiovascular.
                <p style="font-size:1.1em;">
                La probabilidad estimada de riesgo es del <b>{probabilidad_enfermedad:.2%}</b>.
                </p>
                """)
                st.info("""
                El modelo indica que, según tus datos, tu riesgo cardiovascular es **bajo**. ¡Felicidades!
                """)
                st.markdown("""
                **Recomendaciones para mantener un riesgo bajo:**
                * Mantén una alimentación balanceada y rica en frutas y verduras.
                * Continúa realizando actividad física regular.
                * Evita el tabaco y modera el alcohol.
                * Asegúrate de tener un sueño reparador.
                * Realiza chequeos médicos preventivos periódicamente.
                """)
            
            st.expander("Ver Detalles de Probabilidad").write(f"Probabilidad de riesgo detectado por el modelo: {probabilidad_enfermedad:.4f}")

        else:
            st.error("El modelo de predicción no está cargado correctamente. Por favor, verifica la configuración.")



# --- Footer ---
st.markdown("""
<div class="footer">
    <p>
        Desarrollado con por tu equipo de ML | 
        <a href="mailto:contacto@tudominio.com">Contacto</a> | 
        © 2025 Monitor de Salud Cardiovascular. Todos los derechos reservados.
        <br>
        Datos de entrenamiento basados en fuentes públicas (ej. Kaggle CVDP).
    </p>
</div>
""", unsafe_allow_html=True)
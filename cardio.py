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
    /* Fuente general y colores de texto */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; /* Fuente limpia y legible */
        color: #333333;
        background-color: #f0f2f6; /* Un gris claro suave para el fondo */
    }
    .main .block-container {
        padding-top: 2rem;
        padding-right: 3rem; /* Espaciado interno estándar */
        padding-left: 3rem;  /* Espaciado interno estándar */
        padding-bottom: 2rem;
        
        /* *** CAMBIO CLAVE PARA EL ANCHO DEL FORMULARIO Y CONTENIDO PRINCIPAL *** */
        max-width: 1100px; /* Limita el ancho máximo del contenido principal (ajusta este valor) */
        margin: auto; /* Centra el contenedor principal */
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

/* === ESTILOS PARA LA SECCIÓN DE RESULTADOS === */
    /* === ESTILOS PARA LA SECCIÓN DE RESULTADOS === */
    .result-section {
        background-color: #ffffff; /* Fondo blanco para la sección de resultados */
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1); /* Sombra más pronunciada */
        padding: 30px;
        margin-top: 2em;
        text-align: center;
    }
    .result-icon-container {
        display: flex;
        flex-direction: column; /* Cambiado a columna para apilar imagen y leyenda verticalmente */
        justify-content: center;
        align-items: center; /* Centra los elementos a lo largo del eje transversal (horizontalmente para columna) */
        margin-bottom: 0.2em;
        padding-top: 1em;
        height: 100px; /* Altura fija para el contenedor del icono para consistencia */
    }
    /* >>> INICIO DEL CÓDIGO AÑADIDO/MODIFICADO PARA CENTRAR LA IMAGEN <<< */
    [data-testid="stImage"] {
        margin: 0 auto !important; /* Fuerza el centrado del contenedor de la imagen */
        display: block; /* Asegura que ocupe su propio bloque */
    }
    [data-testid="stImage"] p {
        text-align: center;
        width: 100%; /* Asegura que el texto ocupe el ancho completo para centrarse */
    }
    /* >>> FIN DEL CÓDIGO AÑADIDO/MODIFICADO PARA CENTRAR LA IMAGEN <<< */
    .result-icon-container img { /* Estilo específico para las imágenes SVG */
        width: 90px; /* Tamaño fijo para el SVG */
        height: auto; /* Mantiene la proporción */
        display: block; /* Elimina espacio extra debajo de la imagen */
        object-fit: contain; /* Asegura que la imagen se ajuste sin recortarse */
    }
    .result-title-main {
        font-size: 2.2em; /* Título principal del resultado */
        font-weight: bold;
        margin-top: 0.5em;
        margin-bottom: 1em;
    }
    /* Colores específicos para el resultado */
    .risk-low-color { color: #28a745; } /* Verde para bajo riesgo */
    .risk-high-color { color: #dc3545; } /* Rojo para alto riesgo */

    /* Contenedor de tarjetas para las entradas clave */
    .result-card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); /* 3-4 columnas responsivas */
        gap: 20px;
        margin-top: 2em;
        margin-bottom: 2em;
        justify-content: center; /* Centra las tarjetas si no llenan la fila */
    }
    .result-card {
        background-color: #f8f8f8; /* Gris muy claro para el fondo de la tarjeta */
        border-radius: 8px;
        border: 1px solid #e0e0e0; /* Borde sutil */
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05); /* Sombra ligera para las tarjetas */
    }
    .result-card-label {
        font-size: 0.9em;
        color: #666666;
        margin-bottom: 0.3em;
        font-weight: 500;
    }
    .result-card-value {
        font-size: 1.3em;
        font-weight: bold;
        color: #4682B4; /* Azul para los valores */
    }
    .recommendations-box {
        margin-top: 2em;
        padding: 25px;
        background-color: #f8f8f8;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        text-align: left; /* Alinea el texto a la izquierda dentro de las recomendaciones */
    }
    .recommendations-box ul {
        list-style-type: disc; /* Puntos de lista */
        margin-left: 20px;
    }
    .recommendations-box li {
        margin-bottom: 0.5em;
        color: #333333;
    }


</style>""", unsafe_allow_html=True)


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

        genero_display = st.radio("Sexo", options=["Femenino", "Masculino"], index=0)
        genero_encoded = 0 if genero_display == "Femenino" else 1 

        altura = st.number_input("Altura (cm)", min_value=50.0, max_value=250.0, value=170.0, step=1.0)
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
        fuma_input = st.radio("¿Fumas?", options=["No", "Sí"], index=0,)
        fuma_encoded = 0 if fuma_input == "No" else 1

        alcohol_input = st.radio("¿Consumes alcohol?", options=["No", "Sí"], index=0)
        alcohol_encoded = 0 if alcohol_input == "No" else 1

        activo_input = st.radio("¿Realizas actividad física?", options=["No activo", "Activo"], index=1)
        activo_encoded = 0 if activo_input == "No activo" else 1


    submitted = st.form_submit_button("Calcular")

    # --- Lógica de Predicción  ---
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
                'bmi': bmi
            }

            # Convierte el diccionario a un DataFrame de pandas
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

            # Realizar la predicción
            try:
                prediccion_clase = model.predict(processed_input)[0]
                probabilidad_enfermedad = model.predict_proba(processed_input)[:, 1][0]
            except Exception as e:
                st.error(f"Error al realizar la predicción. Detalle: {e}")
                st.stop()


      
            # --- SECCIÓN DE RESULTADOS REDISEÑADA --- 
        
        umbral_decision = 0.5 

        if probabilidad_enfermedad >= umbral_decision: 
            # ALTO RIESGO
            # Contenedor para el icono
            st.markdown('<div class="result-icon-container">', unsafe_allow_html=True)
            st.image("icon-3.svg", width=90)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(f""" 
            <div class="result-title-main risk-high-color"> 
                ¡ALERTA! Tu riesgo es ALTO. 
            </div> 
            <p style="text-align: center; font-size:1.1em; color: #dc3545;"> 
                La probabilidad estimada de riesgo es del <b>{probabilidad_enfermedad:.2%}</b>. 
            </p> 
            """, unsafe_allow_html=True) 

            # Tarjetas con información clave del usuario y el IMC
            st.markdown('<div class="result-card-grid">', unsafe_allow_html=True) 
            st.markdown(f""" 
                <div class="result-card"> 
                    <div class="result-card-label">IMC (kg/m²)</div>
                    <div class="result-card-value">{bmi:.2f}</div>
                </div>
                <div class="result-card"> 
                    <div class="result-card-label">Género</div> 
                    <div class="result-card-value">{genero_display}</div> 
                </div> 
                <div class="result-card"> 
                    <div class="result-card-label">Fumas actualmente</div> 
                    <div class="result-card-value">{fuma_input}</div> 
                </div> 
                <div class="result-card"> 
                    <div class="result-card-label">Nivel de Colesterol</div> 
                    <div class="result-card-value">{colesterol_map[colesterol_input]}</div> 
                </div> 
                <div class="result-card"> 
                    <div class="result-card-label">Presión Sistólica</div> 
                    <div class="result-card-value">{presion_sistolica} mmHg</div> 
                </div> 
                <div class="result-card"> 
                    <div class="result-card-label">Nivel de Glucosa</div> 
                    <div class="result-card-value">{glucosa_map[glucosa_input]}</div> 
                </div> 
                <div class="result-card"> 
                    <div class="result-card-label">Actividad Física</div> 
                    <div class="result-card-value">{activo_input}</div> 
                </div> 
            """, unsafe_allow_html=True) 
            st.markdown('</div>', unsafe_allow_html=True) # Cierre del grid de tarjetas 

            # Recomendaciones 
            st.markdown(""" 
            <div class="recommendations-box"> 
                <p>Basado en la información proporcionada, el modelo sugiere que tienes un <b>riesgo elevado</b>. Te recomendamos enfáticamente que <b>consultes a un profesional de la salud</b> (médico de cabecera o cardiólogo) para una evaluación y asesoramiento personalizado.</p> 
                <p><b>Recomendaciones generales para un riesgo alto:</b></p> 
                <ul> 
                    <li>Monitoriza tu presión arterial y niveles de colesterol/glucosa regularmente.</li> 
                    <li>Adopta una dieta DASH (Dietary Approaches to Stop Hypertension) o mediterránea.</li> 
                    <li>Aumenta tu actividad física con al menos 150 minutos de ejercicio moderado a la semana.</li> 
                    <li>Abandona el hábito de fumar.</li> 
                    <li>Modera o elimina el consumo de alcohol.</li> 
                    <li>Gestiona el estrés de forma activa.</li> 
                </ul> 
            </div> 
            """, unsafe_allow_html=True) 

        else: 
            # BAJO RIESGO 
            # Contenedor para el icono
            st.markdown('<div class="result-icon-container">', unsafe_allow_html=True)
            st.image("icon-1.svg", width=90) 
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(f""" 
            <div class="result-title-main risk-low-color"> 
                ¡Tu riesgo es bajo! 
            </div> 
            <p style="text-align: center; font-size:1.1em; color: #28a745;"> 
                La probabilidad estimada de riesgo es del <b>{probabilidad_enfermedad:.2%}</b>. 
            </p> 
            """, unsafe_allow_html=True) 

            # Tarjetas con información clave del usuario y el IMC
            st.markdown('<div class="result-card-grid">', unsafe_allow_html=True) 
            st.markdown(f""" 
                <div class="result-card"> 
                    <div class="result-card-label">IMC (kg/m²)</div>
                    <div class="result-card-value">{bmi:.2f}</div>
                </div>
                <div class="result-card"> 
                    <div class="result-card-label">Género</div> 
                    <div class="result-card-value">{genero_display}</div> 
                </div> 
                <div class="result-card"> 
                    <div class="result-card-label">Fumas actualmente</div> 
                    <div class="result-card-value">{fuma_input}</div> 
                </div> 
                <div class="result-card"> 
                    <div class="result-card-label">Nivel de Colesterol</div> 
                    <div class="result-card-value">{colesterol_map[colesterol_input]}</div> 
                </div> 
                <div class="result-card"> 
                    <div class="result-card-label">Presión Sistólica</div> 
                    <div class="result-card-value">{presion_sistolica} mmHg</div> 
                </div> 
                <div class="result-card"> 
                    <div class="result-card-label">Nivel de Glucosa</div> 
                    <div class="result-card-value">{glucosa_map[glucosa_input]}</div> 
                </div> 
                <div class="result-card"> 
                    <div class="result-card-label">Actividad Física</div> 
                    <div class="result-card-value">{activo_input}</div> 
                </div> 
            """, unsafe_allow_html=True) 
            st.markdown('</div>', unsafe_allow_html=True) # Cierre del grid de tarjetas 

            # Recomendaciones 
            st.markdown(""" 
            <div class="recommendations-box"> 
                <p>El modelo indica que, según tus datos, tu riesgo cardiovascular es <b>bajo</b>. ¡Felicidades!</p> 
                <p><b>Recomendaciones para mantener un riesgo bajo:</b></p> 
                <ul> 
                    <li>Mantén una alimentación balanceada y rica en frutas y verduras.</li> 
                    <li>Continúa realizando actividad física regular.</li> 
                    <li>Evita el tabaco y modera el alcohol.</li> 
                    <li>Asegúrate de tener un sueño reparador.</li> 
                    <li>Realiza chequeos médicos preventivos periódicamente.</li> 
                </ul> 
            </div> 
            """, unsafe_allow_html=True) 
        
        st.expander("Ver Detalles de Probabilidad").write(f"Probabilidad de riesgo detectado por el modelo: {probabilidad_enfermedad:.4f}") 

        st.markdown('</div>', unsafe_allow_html=True) 

    else: 
        st.error("El modelo de predicción no está cargado correctamente. Por favor, verifica la configuración.") 


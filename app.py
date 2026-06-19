import os
import uuid
import subprocess
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Predicción de riesgo cardiovascular",
    page_icon="🫀",
    layout="centered"
)

st.title("Predicción de riesgo cardiovascular")
st.write(
    "Aplicación conectada a un modelo desplegado en DataRobot para estimar "
    "la probabilidad de enfermedad cardiovascular."
)

st.warning(
    "Uso educativo. Este resultado no reemplaza valoración médica profesional."
)


API_KEY = st.secrets["DATAROBOT_API_KEY"]
DEPLOYMENT_ID = st.secrets["DATAROBOT_DEPLOYMENT_ID"]
HOST = st.secrets["DATAROBOT_HOST"]

PREDICT_SCRIPT = "predict.py"


st.subheader("Datos del paciente")

col1, col2 = st.columns(2)

with col1:
    edad_anios = st.number_input(
        "Edad en años",
        min_value=1,
        max_value=120,
        value=50
    )

    genero = st.selectbox(
        "Género",
        options=[1, 2],
        format_func=lambda x: "1" if x == 1 else "2"
    )

    estatura_cm = st.number_input(
        "Estatura en cm",
        min_value=100,
        max_value=220,
        value=170
    )

    peso_kg = st.number_input(
        "Peso en kg",
        min_value=30.0,
        max_value=250.0,
        value=70.0
    )

    presion_sistolica = st.number_input(
        "Presión sistólica",
        min_value=60,
        max_value=250,
        value=120
    )

    presion_diastolica = st.number_input(
        "Presión diastólica",
        min_value=40,
        max_value=180,
        value=80
    )

with col2:
    colesterol = st.selectbox(
        "Colesterol",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "1 - Normal",
            2: "2 - Alto",
            3: "3 - Muy alto"
        }[x]
    )

    glucosa = st.selectbox(
        "Glucosa",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "1 - Normal",
            2: "2 - Alta",
            3: "3 - Muy alta"
        }[x]
    )

    fuma = st.selectbox(
        "¿Fuma?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Sí"
    )

    consume_alcohol = st.selectbox(
        "¿Consume alcohol?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Sí"
    )

    actividad_fisica = st.selectbox(
        "¿Realiza actividad física?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Sí"
    )


indice_masa_corporal = peso_kg / ((estatura_cm / 100) * (estatura_cm / 100))

st.subheader("Variable calculada")
st.write(f"Índice de masa corporal: **{indice_masa_corporal:.2f}**")


datos_validos = True

if presion_sistolica <= presion_diastolica:
    st.error("La presión sistólica debe ser mayor que la presión diastólica.")
    datos_validos = False


if st.button("Predecir riesgo cardiovascular", disabled=not datos_validos):

    input_data = pd.DataFrame([{
        "genero": genero,
        "estatura_cm": estatura_cm,
        "peso_kg": peso_kg,
        "presion_sistolica": presion_sistolica,
        "presion_diastolica": presion_diastolica,
        "colesterol": colesterol,
        "glucosa": glucosa,
        "fuma": fuma,
        "consume_alcohol": consume_alcohol,
        "actividad_fisica": actividad_fisica,
        "edad_anios": edad_anios,
        "indice_masa_corporal": indice_masa_corporal
    }])

    unique_id = str(uuid.uuid4())
    input_file = f"input_{unique_id}.csv"
    output_file = f"output_{unique_id}.csv"

    input_data.to_csv(input_file, index=False)

    command = [
        "python3",
        PREDICT_SCRIPT,
        input_file,
        output_file,
        DEPLOYMENT_ID,
        f"--api_key={API_KEY}",
        f"--host={HOST}"
    ]

    try:
        with st.spinner("Consultando el modelo desplegado en DataRobot..."):
            subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )

        predictions = pd.read_csv(output_file)

        probabilidad_1 = predictions.loc[0, "enfermedad_cardiovascular_1_PREDICTION"]
        prediccion = predictions.loc[0, "enfermedad_cardiovascular_PREDICTION"]

        st.subheader("Resultado")

        st.metric(
            "Probabilidad de enfermedad cardiovascular",
            f"{probabilidad_1 * 100:.2f}%"
        )

        if int(prediccion) == 1:
            st.error("Predicción: posible enfermedad cardiovascular")
        else:
            st.success("Predicción: no enfermedad cardiovascular")

        st.subheader("Respuesta completa del modelo")
        st.dataframe(predictions)

    except subprocess.CalledProcessError as e:
        st.error("Error ejecutando predict.py")
        st.code(e.stderr)

    except Exception as e:
        st.error("Error inesperado")
        st.code(str(e))

    finally:
        if os.path.exists(input_file):
            os.remove(input_file)

        if os.path.exists(output_file):
            os.remove(output_file)
import gradio as gr
import pandas as pd
import mlflow.sklearn
from gen_ai import explica_prediccion  # Importa la función de explicaciones generativas

# Define las columnas del dataset que usa el modelo
columns = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol"
]

# Carga el modelo registrado en MLflow (reemplaza RUN_ID o usa registro "vino_regressor")
model_uri = "models:/vino_regressor/Production"
model = mlflow.sklearn.load_model(model_uri)

def predict_and_explain(*inputs):
    # Crear dataframe para predicción
    X = pd.DataFrame([inputs], columns=columns)
    pred = model.predict(X)[0]
    
    # Generar explicación con función externa
    explicacion = explica_prediccion(inputs, pred)
    
    # Devuelve la predicción y la explicación para Gradio
    return f"Calidad predicha: {pred:.2f}", explicacion

# Crear interfaz de usuario con Gradio
inputs = [gr.Number(label=col) for col in columns]
outputs = [gr.Textbox(label="Predicción de Calidad"), gr.Textbox(label="Explicación Generativa")]

interface = gr.Interface(
    fn=predict_and_explain,
    inputs=inputs,
    outputs=outputs,
    title="Predicción y Explicación Calidad de Vino Blanco",
    description="Introduce las características químicas y obtén la calidad predicha junto a una explicación automática."
)

if __name__ == "__main__":
    interface.launch()

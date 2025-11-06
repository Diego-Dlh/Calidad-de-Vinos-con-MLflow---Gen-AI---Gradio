import gradio as gr
import pandas as pd
import mlflow.sklearn
from gen_ai import explica_prediccion

columns = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol"
]

model_uri = "runs:/429cce35fef2436f9b6f708831679574/model"
model = mlflow.sklearn.load_model(model_uri)

def predict_and_explain(*inputs):
    X = pd.DataFrame([inputs], columns=columns)
    pred = model.predict(X)[0]
    explicacion = explica_prediccion(inputs, pred)
    return f"Calidad predicha: {pred:.2f}", explicacion

inputs = [gr.Number(label=col) for col in columns]

outputs = [
    gr.Textbox(label="Predicción de Calidad"), 
    gr.Textbox(label="Explicación Generativa", lines=15, max_lines=30, interactive=False)
]

interface = gr.Interface(
    fn=predict_and_explain,
    inputs=inputs,
    outputs=outputs,
    title="Predicción y Explicación Calidad de Vino Blanco",
    description="Introduce las características químicas y obtén la calidad predicha junto a una explicación automática."
)

if __name__ == "__main__":
    interface.launch()

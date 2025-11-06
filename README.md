# Predicción y Generación de Insights sobre Calidad de Vinos

## Descripción

Este proyecto implementa un pipeline completo de MLOps para predecir la calidad de vinos blancos usando un modelo clásico de aprendizaje automático y una interfaz interactiva con generación automática de explicaciones mediante IA generativa usando Ollama con el modelo `phi3:mini`.

Se usa MLflow para el tracking y versionamiento de experimentos, Gradio para la interfaz de usuario, y Ollama con modelo `phi3:mini` para explicar de forma automática las predicciones.

---

## Estructura del proyecto
<pre>
ProyectoVino/
│
├── data/ # Dataset Wine Quality
│ └── winequality-white.csv
├── notebooks/
│ └── exploracion.ipynb # Exploración y análisis de datos
├── project/
│ ├── MLproject # Configuración experimentos MLflow
│ ├── train.py # Script entrenamiento y registro
│ ├── conda.yaml # Entorno de conda
├── gradio_app/
│ ├── app.py # Interfaz web con Gradio
│ ├── gen_ai.py # Generación de explicaciones con Ollama phi3:mini
├── README.md # Este archivo
</pre>
---

## Requisitos

- Python 3.9+
- Ollama instalado y ejecutándose localmente
- Modelo `phi3:mini` descargado en Ollama (`ollama pull phi3:mini`)
- Paquetes Python: mlflow, gradio, pandas, scikit-learn, ollama (instalación en conda.yaml o pip)

---

## Instalación y ejecución

1. Clona el repositorio y entra al directorio del proyecto.
2. Crea y activa un ambiente virtual:
3. Instala dependencias:
4. Asegúrate de tener Ollama instalado y el modelo `phi3:mini` descargado:
Y ejecutando localmente antes de iniciar la app.
5. Ejecuta experimentos MLflow:
6. Ejecuta MLflow UI para revisar resultados:
7. Ejecuta la app Gradio para predicción y explicación:


---

## Uso

Introduce las características químicas del vino en la interfaz web para obtener:

- Predicción de la calidad del vino.
- Explicación automática generada mediante Ollama `phi3:mini`.

---

## Reflexiones Éticas

- Se debe considerar la responsabilidad en la interpretación de las explicaciones generadas automáticamente.
- La calidad de la predicción depende de la representación y diversidad del dataset.
- La IA generativa puede introducir sesgos; se recomienda análisis crítico de los resultados.

---

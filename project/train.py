# train.py

import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def main(data_path, n_estimators, max_depth, random_state):
    # Leer el dataset
    data = pd.read_csv(data_path)
    X = data.drop("quality", axis=1)
    y = data["quality"]

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Configurar el experimento MLflow
    mlflow.set_experiment("vino-quality")

    with mlflow.start_run():
        # Entrenar modelo
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)

        # Hacer predicciones
        predictions = model.predict(X_test)

        # Calcular métricas
        mse = mean_squared_error(y_test, predictions)

        # Registrar parámetros, métricas y modelo en MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("mse", mse)

        mlflow.sklearn.log_model(model, "model", registered_model_name="vino_regressor")

        print(f"Modelo registrado con MSE: {mse}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data/winequality-white.csv", help="Ruta al archivo CSV de datos")
    parser.add_argument("--n_estimators", type=int, default=100, help="Número de árboles")
    parser.add_argument("--max_depth", type=int, default=None, help="Profundidad máxima del árbol")
    parser.add_argument("--random_state", type=int, default=42, help="Semilla aleatoria")
    args = parser.parse_args()
    main(args.data, args.n_estimators, args.max_depth, args.random_state)

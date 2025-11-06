import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Configuración del experimento
mlflow.set_experiment("vino-quality")
mlflow.autolog()  # Opcional: registra mucho automáticamente

data = pd.read_csv("data/winequality-white.csv")
X = data.drop("quality", axis=1)
y = data["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    params = {"n_estimators": 100, "random_state": 42}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    mlflow.log_params(params)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model", registered_model_name="vino_regressor")

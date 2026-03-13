import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Load dataset
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start MLflow run
with mlflow.start_run():
    # Define model
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Predict + evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Log params + metrics
    mlflow.log_param("n_estimators", 50)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", acc)

    # Log model
    mlflow.sklearn.log_model(model, "model")

print("Training complete.")

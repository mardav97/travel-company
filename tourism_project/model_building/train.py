import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, accuracy_score,precision_score,f1_score
# for model serialization
import joblib
# mlflow libraries
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature

MODEL_PATH = 'tourism_project/deployment/best_model.pkl'
xtrain = pd.read_csv('tourism_project/data/xtrain.csv')
xtest = pd.read_csv('tourism_project/data/xtest.csv')
ytrain = pd.read_csv('tourism_project/data/ytrain.csv')
ytest = pd.read_csv('tourism_project/data/ytest.csv')

signature = infer_signature(xtrain, ytrain)

model = XGBClassifier(random_state=42, n_jobs=-1)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="recall",
    n_jobs=-1
)


mlflow.set_experiment("tourism_model_training")

with mlflow.start_run():

    # Tune model
    grid_search.fit(xtrain, ytrain)

    best_model = grid_search.best_estimator_

    # Predictions
    y_pred = best_model.predict(xtest)

    # Evaluation metrics
    accuracy = accuracy_score(ytest, y_pred)
    precision = precision_score(ytest, y_pred)
    recall = recall_score(ytest, y_pred)
    f1 = f1_score(ytest, y_pred)

    # Log best hyperparameters
    mlflow.log_params(grid_search.best_params_)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log model to MLflow
    # mlflow.xgboost.log_model(
    #     xgb_model = best_model,
    #     name="model",
    #     signature = signature,
    #     input_example=xtrain[:5],
    #     model_format="json"
    # )

    print("Best parameters:", grid_search.best_params_)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # --------------------------------------------------
    # 5. Save model for deployment
    # --------------------------------------------------

    os.makedirs("tourism_project/deployment", exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)

    print(f"\nBest model saved to: {MODEL_PATH}")

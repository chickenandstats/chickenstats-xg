### Script for running mlflow experiments and logging them to mlflow on GCP ###

## Import dependencies

import argparse

import pandas as pd

import numpy as np

from pathlib import Path

import os

import xgboost as xgb

import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)

import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature

import optuna

import warnings

from dotenv import load_dotenv

import matplotlib.pyplot as plt

from yellowbrick.classifier import (
    ClassificationReport,
    ClassPredictionError,
    ROCAUC,
    PrecisionRecallCurve,
    ConfusionMatrix,
)
from yellowbrick.model_selection import FeatureImportances

import shap

import seaborn as sns

import cronitor

## Functions

def model_metrics(y, y_pred, y_pred_proba):
    """Returns various model metrics for a tuned model"""

    metric_names = [
        "accuracy",
        "average_precision",
        "balanced_accuracy",
        "f1",
        "f1_weighted",
        "precision",
        "precision_weighted",
        "recall",
        "recall_weighted",
        "roc_auc",
        "log_loss",
    ]

    accuracy = sklearn.metrics.accuracy_score(y, y_pred)

    average_precision = sklearn.metrics.average_precision_score(y, y_pred_proba)

    balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y, y_pred)

    f1 = sklearn.metrics.f1_score(y, y_pred, zero_division=0)

    f1_weighted = sklearn.metrics.f1_score(y, y_pred, average="weighted", zero_division=0)

    precision = sklearn.metrics.precision_score(y, y_pred, zero_division=0)

    precision_weighted = sklearn.metrics.precision_score(y, y_pred, average="weighted", zero_division=0)

    recall = sklearn.metrics.recall_score(y, y_pred, zero_division=0)

    recall_weighted = sklearn.metrics.recall_score(y, y_pred, average="weighted", zero_division=0)

    roc_auc = sklearn.metrics.roc_auc_score(y, y_pred_proba)

    log_loss = sklearn.metrics.log_loss(y, y_pred_proba)

    metrics = [
        accuracy,
        average_precision,
        balanced_accuracy,
        f1,
        f1_weighted,
        precision,
        precision_weighted,
        recall,
        recall_weighted,
        roc_auc,
        log_loss,
    ]

    model_metrics = dict(zip(metric_names, metrics))

    return model_metrics


def model_viz(model, X_train, y_train, X_test, y_test):
    """Function to generate model visualizations"""

    dpi = 100
    figsize = (6, 4)

    encoder = {0: "no goal", 1: "goal"}

    ## Classification report

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)

    viz = ClassificationReport(model, encoder=encoder, support=True, cmap="RdPu", ax=ax)

    viz.fit(X_train, y_train)

    viz.score(X_test, y_test)

    viz.finalize()

    classification_report = fig

    ## ROC-AUC

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)

    viz = ROCAUC(model, encoder=encoder)

    viz.fit(X_train, y_train)

    viz.score(X_test, y_test)

    viz.finalize()

    roc_auc = fig

    ## Class Prediction Errors

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)

    viz = ClassPredictionError(model, encoder=encoder)

    viz.fit(X_train, y_train)

    viz.score(X_test, y_test)  # Evaluate the model on the test data

    viz.finalize()

    class_prediction = fig

    ## Precision Recall Curve

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)

    viz = PrecisionRecallCurve(model, encoder=encoder)

    viz.fit(X_train, y_train)

    viz.score(X_test, y_test)  # Evaluate the model on the test data

    viz.finalize()

    precision_recall = fig

    ## Feature Importance

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)

    viz = FeatureImportances(
        model,
        relative=False,
        ax=ax,
        topn=10,
    )

    viz.fit(X_train, y_train)

    viz.score(X_test, y_test)  # Evaluate the model on the test data

    viz.finalize()

    importance = fig

    ## Relative Feature Importance

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)

    viz = FeatureImportances(model, relative=True, topn=10, ax=ax)

    viz.fit(X_train, y_train)

    viz.score(X_test, y_test)  # Evaluate the model on the test data

    viz.finalize()

    relative_importance = fig

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)

    viz = ConfusionMatrix(model, cmap="RdPu", encoder=encoder)

    viz.fit(X_train, y_train)

    viz.score(X_test, y_test)  # Evaluate the model on the test data

    viz.finalize()

    confusion_matrix = fig

    return (
        classification_report,
        roc_auc,
        class_prediction,
        precision_recall,
        importance,
        relative_importance,
        confusion_matrix,
    )


def log_viz(
    classification_report,
    roc_auc,
    class_prediction,
    precision_recall,
    importance,
    relative_importance,
    confusion_matrix,
):
    mlflow.log_figure(classification_report, "viz/classification_report.png")

    classification_report.clear()

    mlflow.log_figure(roc_auc, "viz/roc_auc.png")

    roc_auc.clear()

    mlflow.log_figure(class_prediction, "viz/class_prediction_error.png")

    class_prediction.clear()

    mlflow.log_figure(precision_recall, "viz/precision_recall_curve.png")

    precision_recall.clear()

    mlflow.log_figure(importance, "viz/feature_importance.png")

    importance.clear()

    mlflow.log_figure(relative_importance, "viz/relative_feature_importance.png")

    relative_importance.clear()

    mlflow.log_figure(confusion_matrix, "viz/confusion_matrix.png")

    confusion_matrix.clear()


def load_data(model_name):
    # Folder for saving the files
    SAVE_FOLDER = Path("./data/processed/")

    filepath = SAVE_FOLDER / f"{model_name}.csv"

    df = pd.read_csv(filepath, index_col=0)

    train = df.loc[df.season != 20212022].drop("season", axis=1)
    test = df.loc[df.season == 20212022].drop("season", axis=1)

    X_train = train.drop("goal", axis=1)
    y_train = train["goal"]

    X_test = test.drop("goal", axis=1)
    y_test = test["goal"]

    scale_pos_weight = y_train.loc[y_train == 0].count() / y_train.loc[y_train == 1].count()

    if model_name == "empty_against":

        scale_pos_weight = 1

    return X_train, X_test, y_train, y_test, scale_pos_weight


def objective(trial):
    """Objective function that will be used as part of Optuna"""

    with mlflow.start_run(run_id=parent_info.run_id):
        with mlflow.start_run(nested=True) as current_run:
            params = {
                "objective": "binary:logistic",
                #'eval_metric': ['auc', 'log_loss'],
                "verbosity": 0,
                "random_state": 615,
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
                "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
                "scale_pos_weight": trial.suggest_int(
                    "scale_pos_weight", 1, scale_pos_weight
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-8, 1.0, log=True
                ),
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.4, 1.0, step=0.05),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.4, 1.0, step=0.05
                ),
            }

            run_data = current_run.info

            MONITOR.ping(
                env="training",
                message=f"Starting {run_data.run_name} ({parent_info.run_name} - {study_name})...",
                host="macstudio",
            )

            experiment_id = run_data.experiment_id

            mlflow.log_params(params)

            params.update({"eval_metric": ["auc", "logloss"]})

            model = xgb.XGBClassifier(**params)

            SEED = 615

            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

            evals = cross_validate(
                model,
                X_train,
                y_train,
                scoring=[
                    "roc_auc",
                    "precision",
                    "recall",
                    "f1",
                    "accuracy",
                    "neg_log_loss",
                ],
                cv=kfold,
                n_jobs=-1,
            )

            metrics = {}

            for metric, results in evals.items():
                metric = metric.replace("test_", "")

                if metric == "neg_log_loss":
                    metric = metric.replace("neg_", "")

                    results = [x * -1 for x in results]

                metrics.update(
                    {
                        f"train_{metric}_mean": np.mean(results),
                        f"train_{metric}_std": np.std(results),
                    }
                )

            mlflow.log_metrics(metrics)

            evals = pd.DataFrame(evals)

            evals["kfold"] = evals.index + 1

            evals = evals.set_index("kfold")

            evals = evals.to_html(na_rep="", float_format=lambda x: str(round(x, 3)))

            mlflow.log_text(evals, "performance/train_cross_validation.html")

            model.fit(X_train, y_train)

            y_preds = model.predict(X_test)

            y_probs = model.predict_proba(X_test)[:, 1]

            # Logging model signature, class, and name
            signature = infer_signature(X_test, y_preds)

            mlflow.xgboost.log_model(model, "model", signature=signature)

            test_metrics = model_metrics(y_test, y_preds, y_probs)

            metrics = {}

            for metric, result in test_metrics.items():
                metrics.update(
                    {
                        f"test_{metric}": np.mean(result),
                    }
                )

            mlflow.log_metrics(metrics)

            if metrics["test_roc_auc"] < 0.5 or metrics["test_precision"] == 0:
                performance_tag = "none"

            elif metrics["test_roc_auc"] < 0.75:
                performance_tag = "low"

            elif metrics["test_roc_auc"] < 0.78:
                performance_tag = "medium"

            elif metrics["test_roc_auc"] <= 0.8:
                performance_tag = "high"

            else:
                performance_tag = "very high"

            tags = {
                "performance": performance_tag,
                "experiment_name": study_name,
                "experiment_id": experiment_id,
                "estimator_name": model.__class__.__name__,
                "estimator_class": model.__class__,
                "parent_id": parent_info.run_id,
                "parent_name": parent_info.run_name,
                "level": "child",
            }

            mlflow.set_tags(tags)

            class_report = sklearn.metrics.classification_report(
                y_test,
                y_preds,
                labels=[0, 1],
                target_names=["no goal", "goal"],
                output_dict=True,
            )

            class_report = pd.DataFrame(class_report).to_html(
                na_rep="", float_format=lambda x: str(round(x, 3))
            )

            mlflow.log_text(class_report, "performance/test_classification_report.html")

            (
                classification_report,
                roc_auc,
                class_prediction,
                precision_recall,
                importance,
                relative_importance,
                confusion_matrix,
            ) = model_viz(model, X_train, y_train, X_test, y_test)

            log_viz(
                classification_report,
                roc_auc,
                class_prediction,
                precision_recall,
                importance,
                relative_importance,
                confusion_matrix,
            )

            return metrics["test_roc_auc"], metrics["test_log_loss"], metrics["test_f1"]


def tune_model(model_name, version, storage, max_trials, run=None):
    """Wraps all of the over tuning functions into one"""

    global X_train, X_test, y_train, y_test, scale_pos_weight

    X_train, X_test, y_train, y_test, scale_pos_weight = load_data(model_name)

    global study_name, experiment_id

    study_name = f"{model_name}-{version}"

    EXPERIMENT = mlflow.set_experiment(study_name)

    experiment_id = EXPERIMENT.experiment_id

    global parent_info

    tags = {
        "experiment_name": study_name,
        "experiment_id": experiment_id,
        "level": "parent",
    }

    if run == None:
        with mlflow.start_run(tags=tags) as parent_run:
            parent_info = parent_run.info

            MONITOR.ping(
                env="training",
                message=f"Starting {parent_info.run_name} ({study_name})...",
                host="macbook",
            )

    else:
        with mlflow.start_run(run_id=run) as parent_run:
            parent_info = parent_run.info

            MONITOR.ping(
                env="training",
                message=f"Starting {parent_info.run_name} ({study_name})...",
                host="macbook",
            )

    try:
        study = optuna.create_study(
            study_name=study_name,
            load_if_exists=True,
            storage=storage,
            directions=["maximize", "minimize", "maximize"],
        )

    except optuna.exceptions.StorageInternalError:
        study = optuna.load_study(study_name=study_name, storage=storage)

    study.optimize(objective, n_trials=max_trials, show_progress_bar=True)

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="xG Training", description="Python script for training xG model"
    )
    parser.add_argument("--strength", "-s", type=str, required=True)
    parser.add_argument("--version", "-v", type=str, required=True)
    parser.add_argument("--run", "-r", type=str, required=False)
    parser.add_argument("--trials", "-t", type=int, required=False)
    parser.add_argument("--delete", "-d", type=bool, required=False)
    args = parser.parse_args()

    strengths_list = [
        "even_strength",
        "powerplay",
        "shorthanded",
        "empty_for",
        "empty_against",
    ]

    if args.strength not in strengths_list:
        raise Exception(
            "Strength name is not supported, try 'even_strength', 'powerplay', 'shorthanded', 'empty_for', or 'empty_against'"
        )

    ## Setting seaborn style for graphs

    sns.set_style("white")

    ## Ignoring warnings

    warnings.filterwarnings("ignore")
    warnings.filterwarnings(action="ignore", module="mlflow.models.model")

    ## Setting environment variables

    load_dotenv()

    CRONITOR_KEY = os.environ.get("CRONITOR_KEY")
    CRONITOR_JOB = os.environ.get("CRONITOR_JOB")

    OPTUNA_STORAGE = os.environ.get("OPTUNA_STORAGE")

    cronitor.api_key = CRONITOR_KEY

    global MONITOR

    MONITOR = cronitor.Monitor(CRONITOR_JOB)

    model_name = args.strength

    version = args.version

    if args.trials is None:
        trials = 100

    else:
        trials = args.trials

    if args.run is None:
        run = None

    else:
        run = args.run

    db_host = os.environ["DB_HOST"]
    db_user = os.environ["DB_USER"]
    db_password = os.environ["DB_PASSWORD"]
    db_name = os.environ["DB_NAME"]
    mlflow_tracking_uri = mlflow.get_tracking_uri()

    postgres_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}/{db_name}"

    storage = optuna.storages.RDBStorage(url=postgres_url, skip_compatibility_check=True)

    if args.delete:
        study_name = f"{model_name}-{version}"

        optuna.delete_study(study_name, storage=storage)

    study = tune_model(
        model_name=model_name,
        version=version,
        storage=storage,
        max_trials=trials,
        run=run,
    )

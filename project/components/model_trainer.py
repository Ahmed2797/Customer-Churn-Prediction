import sys
import os
from typing import Tuple, List
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from neuro_mf import ModelFactory
from project.logger import logging
from project.exception import CustomException
from project.entity.artifacts import (
    Data_Transformation_Artifact,
    Model_Trainer_Artifact,
    ClassificationMetricArtifact
)
from project.entity.config import Model_Trainer_Config
from project.entity.estimator import ProjectModel
from project.utils import load_numpy_array, load_object, save_object
import joblib
import mlflow
import mlflow.sklearn
# import dagshub

# # dagshub.init(repo_owner='Ahmed2797', repo_name='Churn-Prediction', mlflow=True)


class Model_Trainer:
    """
    Model_Trainer handles training, evaluating, selecting, saving, and logging ML models.

    Attributes:
        data_transformation_artifact (Data_Transformation_Artifact): Paths/objects from the data transformation stage.
        model_trainer_config (Model_Trainer_Config): Configuration for training, saving, and MLflow logging.
    """

    def __init__(self, 
                 data_transformation_artifact: Data_Transformation_Artifact,
                 model_trainer_config: Model_Trainer_Config):
        """
        Initialize Model_Trainer with required artifacts and configuration.

        Args:
            data_transformation_artifact (Data_Transformation_Artifact): Outputs from data transformation.
            model_trainer_config (Model_Trainer_Config): Configuration for training, saving, and logging.
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def track_mlflow(self, model_path: str, metrics: dict):
        """
        Log metrics and model to MLflow.

        Args:
            model_path (str): Path of the trained model file.
            metrics (dict): Dictionary of metrics (Accuracy, Precision, Recall, F1).

        Raises:
            CustomException: If MLflow logging fails.
        """
        try:
            mlflow.set_tracking_uri(self.model_trainer_config.mlflow_tracking_uri)
            mlflow.set_experiment(self.model_trainer_config.mlflow_experiment_name)

            with mlflow.start_run():
                # Log metrics
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)
                # Log model file
                mlflow.log_artifact(model_path)
                # Log model object to MLflow
                try:
                    model_obj = joblib.load(model_path)
                    mlflow.sklearn.log_model(model_obj, artifact_path="model")
                except Exception as e:
                    logging.warning(f"Failed to log model object to MLflow: {e}")

        except Exception as e:
            raise CustomException(e, sys)

    def get_all_models_metrics(self, x_train, y_train, x_test, y_test) -> Tuple[List[dict], object, dict]:
        """
        Train all models from configuration, evaluate them, and select the best based on F1 score.

        Args:
            x_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            x_test (np.ndarray): Testing features.
            y_test (np.ndarray): Testing labels.

        Returns:
            Tuple[List[dict], object, dict]: 
                - List of metrics for all trained models.
                - Best model object.
                - Metrics dictionary of best model.

        Raises:
            CustomException: If training or evaluation fails.
        """
        try:
            logging.info("Initializing ModelFactory and training all models...")
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.param_yaml)
            model_list = model_factory.get_initialized_model_list()

            # Grid search all models
            grid_searched_models = model_factory.initiate_best_parameter_search_for_initialized_models(
                initialized_model_list=model_list,
                input_feature=x_train,
                output_feature=y_train
            )

            results = []

            for grid_model in grid_searched_models:
                model = grid_model.best_model
                model_name = type(model).__name__
                y_pred = model.predict(x_test)

                metrics = {
                    "Model": model_name,
                    "ModelObject": model,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                    "ClassificationReport": classification_report(y_test, y_pred, zero_division=0)
                }
                results.append(metrics)

            # Select best model by F1 score
            best_result = max(results, key=lambda x: x["F1"])
            best_model_obj = best_result["ModelObject"]
            best_model_metrics = {
                "Accuracy": best_result["Accuracy"],
                "Precision": best_result["Precision"],
                "Recall": best_result["Recall"],
                "F1": best_result["F1"]
            }

            logging.info(f"Best Model: {best_result['Model']} | F1 Score: {best_result['F1']:.4f}")
            return results, best_model_obj, best_model_metrics

        except Exception as e:
            raise CustomException(e, sys)

    def init_model(self) -> Model_Trainer_Artifact:
        """
        Main method to:
            1. Train models
            2. Evaluate and pick the best
            3. Save the model and pipeline
            4. Optionally log metrics to MLflow

        Returns:
            Model_Trainer_Artifact: Contains paths and metrics of the trained model.

        Raises:
            CustomException: If training, evaluation, saving, or logging fails.
        """
        try:
            logging.info("Starting model training process...")

            # Load train/test arrays
            train_arr = load_numpy_array(self.data_transformation_artifact.transform_train_path)
            test_arr = load_numpy_array(self.data_transformation_artifact.transform_test_path)
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Train models & get metrics
            best_result, best_model_obj, best_model_metrics = self.get_all_models_metrics(
                x_train, y_train, x_test, y_test
            )

            print(best_model_metrics)
            # Print all metrics
            for m in best_result:
                print("="*50)
                print(f"Model: {m['Model']}")
                print(f"Accuracy : {m['Accuracy']:.4f}")
                print(f"Precision: {m['Precision']:.4f}")
                print(f"Recall   : {m['Recall']:.4f}")
                print(f"F1 Score : {m['F1']:.4f}")
                print("Classification Report:")
                print(m["ClassificationReport"])

            # Load preprocessing object
            preprocessor_obj = load_object(self.data_transformation_artifact.preprocessing_pkl)

            # Wrap model with preprocessing pipeline
            prediction_model = ProjectModel(
                transform_object=preprocessor_obj,
                best_model_details=best_model_obj
            )
            print("Final Prediction Model",prediction_model)
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.model_trainer_config.best_model_object), exist_ok=True)

            # Save final pipeline and model
            # save_object(self.model_trainer_config.best_model_object, prediction_model)
            save_object(self.model_trainer_config.final_model_path, prediction_model)

            # Save raw model object separately
            best_model_path = os.path.join(self.model_trainer_config.model_train_dir, "best_model.pkl")
            save_object(best_model_path, best_model_obj)

            # Optionally log metrics & model to MLflow
            # self.track_mlflow(best_model_path, best_model_metrics)

            # Create metric artifact
            y_pred_best = best_model_obj.predict(x_test)
            metrics_artifact = ClassificationMetricArtifact(
                accuracy_score=accuracy_score(y_test, y_pred_best),
                f1_score=f1_score(y_test, y_pred_best, average='weighted', zero_division=0),
                precision_score=precision_score(y_test, y_pred_best, average='weighted', zero_division=0),
                recall_score=recall_score(y_test, y_pred_best, average='weighted', zero_division=0)
            )

            logging.info(f"Final metrics: {metrics_artifact}")
            logging.info("Model training and saving completed successfully.")

            return Model_Trainer_Artifact(
                trained_model_file_path=self.model_trainer_config.final_model_path,
                metric_artifact=metrics_artifact
            )


        except Exception as e:
            raise CustomException(e, sys)




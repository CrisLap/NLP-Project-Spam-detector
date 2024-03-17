import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier



def create_tfidf(trial):
    """
    Create a TF-IDF vectorizer with parameters optimized using Optuna's trial object.

    Args:
        trial: Optuna trial object.

    Returns:
        TfidfVectorizer: TF-IDF vectorizer with optimized parameters.
    """
    tfidf_max_df = trial.suggest_float("tfidf_max_df", 0.8, 0.99)
    tfidf_min_df = trial.suggest_float("tfidf_min_df", 0.01, 0.15)
    return TfidfVectorizer(min_df=tfidf_min_df, max_df=tfidf_max_df)

def create_lr(trial):
    """
    Create a Logistic Regression classifier with parameters optimized using Optuna's trial object.

    Args:
        trial: Optuna trial object.

    Returns:
        LogisticRegression: Logistic Regression classifier with optimized parameters.
    """
    lr_class_weight = trial.suggest_categorical("lr_class_weight", [None, "balanced"])
    lr_c = trial.suggest_float("lr_c", 1e-2, 1e2, log=True)
    return LogisticRegression(C=lr_c,
                              class_weight=lr_class_weight)

def create_nb(trial):
    """
    Create a Naive Bayes classifier (either MultinomialNB or ComplementNB) with parameters optimized using Optuna's trial object.

    Args:
        trial: Optuna trial object.

    Returns:
        MultinomialNB or ComplementNB: Naive Bayes classifier with optimized parameters.
    """
    nb_classifier_type = trial.suggest_categorical("nb_classifier_type", ["multinomial", "complement"])
    nb_alpha = trial.suggest_float("nb_alpha", 1e-4, 1, log=True)
    if nb_classifier_type == "multinomial":
        classifier_obj = MultinomialNB(alpha=nb_alpha)
    else:
        classifier_obj = ComplementNB(alpha=nb_alpha)
    return classifier_obj

def create_svc(trial):
    """
    Create a Support Vector Classifier with parameters optimized using Optuna's trial object.

    Args:
        trial: Optuna trial object.

    Returns:
        SVC: Support Vector Classifier with optimized parameters.
    """
    svc_class_weight = trial.suggest_categorical("svc_class_weight", [None, "balanced"])
    svc_c = trial.suggest_float("svc_c", 1e-2, 1e2, log=True)
    svc_kernel = trial.suggest_categorical("svc_kernel", ["poly", "rbf", "sigmoid", "linear"])
    return SVC(C=svc_c,
               kernel=svc_kernel,
               class_weight=svc_class_weight)

def create_rf(trial):
    """
    Create a Random Forest classifier with parameters optimized using Optuna's trial object.

    Args:
        trial: Optuna trial object.

    Returns:
        RandomForestClassifier: Random Forest classifier with optimized parameters.
    """
    rf_n_estimators = trial.suggest_int("rf_n_estimators", 20, 200)
    rf_max_features = trial.suggest_categorical("rf_max_features", [None, "sqrt"])
    rf_class_weight = trial.suggest_categorical("rf_class_weight", [None, "balanced"])
    return RandomForestClassifier(n_estimators=rf_n_estimators,
                                  max_features=rf_max_features,
                                  class_weight=rf_class_weight,
                                  random_state=0)

def create_mlp(trial):
    """
    Create a Multi-layer Perceptron classifier with parameters optimized using Optuna's trial object.

    Args:
        trial: Optuna trial object.

    Returns:
        MLPClassifier: Multi-layer Perceptron classifier with optimized parameters.
    """
    n_layers = trial.suggest_int("n_layers", 1, 2)
    mlp_activation = trial.suggest_categorical("mlp_activation", ["tanh", "relu"])
    mlp_alpha = trial.suggest_float("mlp_alpha", 1e-5, 1e-1, log=True)
    
    mlp_layers = []
    for i in range(n_layers):
        mlp_layers.append(trial.suggest_int("n_units l{}".
            format(i), 32, 256))

    return MLPClassifier(
        hidden_layer_sizes=tuple(mlp_layers),
        activation=mlp_activation,
        alpha=mlp_alpha,
        solver="adam",
        max_iter=150,
        tol=0.001,
        random_state=0
    )


class Objective():
    """
    Objective function for hyperparameter optimization using Optuna.

    This class defines an objective function to be optimized by Optuna. It evaluates the performance of different classifiers 
    (Logistic Regression, Naive Bayes, Support Vector Classifier, Random Forest, Multi-layer Perceptron) with TF-IDF vectorization 
    on a given dataset using a 5-fold stratified cross-validation approach.

    Args:
        X (array-like): Feature matrix.
        y (array-like): Target vector.

    Returns:
        float: Mean F1 score of the classifiers across 5-fold cross-validation.
    """
    def __init__(self, X, y):
        """
        Initialize the Objective function.

        Args:
            X (array-like): Feature matrix.
            y (array-like): Target vector.
        """
        self.X = X
        self.y = y
        

    def __call__(self, trial):
        """
        Call method to evaluate the objective function.

        Args:
            trial: Optuna trial object.

        Returns:
            float: Mean F1 score of the classifiers across 5-fold cross-validation.
        """
        X, y = self.X, self.y
        classifiers = ["lr", "nb", "svc", "rf", "mlp"]
        
        tfidf_vectorizer = create_tfidf(trial)
        classifier_name = trial.suggest_categorical("classifier", classifiers)
        if classifier_name == "lr":
            classifier_obj = create_lr(trial)
        elif classifier_name == "nb":
            classifier_obj = create_nb(trial)
        elif classifier_name == "svc":
            classifier_obj = create_svc(trial)
        elif classifier_name == "rf":
            classifier_obj = create_rf(trial)
        else:
            classifier_obj = create_mlp(trial)
        
        pipeline = Pipeline([
            ('tfidf', tfidf_vectorizer),
            ('model', classifier_obj)
        ])
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        f1_scores = np.empty(5)

        for i, (train_index, val_index) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            pipeline.fit(X_train, y_train)
            y_val_pred = pipeline.predict(X_val)
            f1_scores[i] = f1_score(y_val, y_val_pred)
        
        print(f1_scores)
        f1 = f1_scores.mean()
        return f1
    

def get_best_pipeline(parameters):
    """
    Construct and return a pipeline with the best parameters for text classification.

    Args:
        parameters (dict): A dictionary containing the parameters for constructing the pipeline.
            The dictionary should include the following keys:
                - "tfidf_min_df" (float): Minimum document frequency for TF-IDF vectorization.
                - "tfidf_max_df" (float): Maximum document frequency for TF-IDF vectorization.
                - "classifier" (str): Type of classifier to use ("lr", "nb", "svc", "rf", "mlp").
                - Additional keys depending on the chosen classifier:
                    - For Logistic Regression ("lr"):
                        - "lr_c" (float): Regularization parameter.
                        - "lr_class_weight" (str or None): Class weights for imbalanced classes.
                    - For Naive Bayes ("nb"):
                        - "nb_classifier_type" (str): Type of Naive Bayes classifier ("multinomial" or "complement").
                        - "nb_alpha" (float): Additive smoothing parameter.
                    - For Support Vector Classifier ("svc"):
                        - "svc_c" (float): Regularization parameter.
                        - "svc_kernel" (str): Kernel type for SVC ("poly", "rbf", "sigmoid", "linear").
                        - "svc_class_weight" (str or None): Class weights for imbalanced classes.
                    - For Random Forest ("rf"):
                        - "rf_n_estimators" (int): Number of trees in the forest.
                        - "rf_max_features" (str or None): Maximum number of features considered for splitting.
                        - "rf_class_weight" (str or None): Class weights for imbalanced classes.
                    - For Multi-layer Perceptron ("mlp"):
                        - "n_layers" (int): Number of hidden layers.
                        - "n_units l0" (int): Number of units in the first hidden layer.
                        - "n_units l1" (int): Number of units in the second hidden layer (if applicable).
                        - "mlp_activation" (str): Activation function for hidden layers ("logistic" or "relu").
                        - "mlp_alpha" (float): L2 penalty parameter.

    Returns:
        Pipeline: A scikit-learn pipeline consisting of TF-IDF vectorization and the specified classifier.

    """
    
    tfidf_vectorizer = TfidfVectorizer(
        min_df=parameters["tfidf_min_df"],
        max_df=parameters["tfidf_max_df"]
    )
    if parameters["classifier"] == "lr":
        model = LogisticRegression(C=parameters["lr_c"],
                                   class_weight=parameters["lr_class_weight"])
    elif parameters["classifier"] == "nb":
        if parameters["nb_classifier_type"] == "multinomial":
            model = MultinomialNB(alpha=parameters["nb_alpha"])
        else:
            model = ComplementNB(alpha=parameters["nb_alpha"])
    elif parameters["classifier"] == "svc":
        model = SVC(
            C=parameters["svc_c"],
            kernel=parameters["svc_kernel"],
            class_weight=parameters["svc_class_weight"]
        )
    elif parameters["classifier"] == "rf":
        model = RandomForestClassifier(
            n_estimators=parameters["rf_n_estimators"],
            max_features=parameters["rf_max_features"],
            class_weight=parameters["rf_class_weight"],
            random_state=0
        )
    else:
        if parameters["n_layers"] == 1:
            model = MLPClassifier(
                hidden_layer_sizes=(parameters["n_units l0"],),
                activation=parameters["mlp_activation"],
                alpha=parameters["mlp_alpha"],
                solver="adam",
                max_iter=100,
                tol=0.001,
                random_state=0
            )
        else:
            model = MLPClassifier(
                hidden_layer_sizes=(parameters["n_units l0"],parameters["n_units l1"]),
                activation=parameters["mlp_activation"],
                alpha=parameters["mlp_alpha"],
                solver="adam",
                max_iter=100,
                tol=0.001,
                random_state=0
            )
        
    return Pipeline([
        ('tfidf', tfidf_vectorizer),
        ('model', model)
    ])


def plot_confusion_matrix(model, data: tuple, title="Train", subplot=121):
    """
    Plot confusion matrix for a classification model.

    Args:
        model: 
            The classification model.
        data (tuple): 
            Tuple containing X and y.
        title (str, optional): 
            Title for the confusion matrix plot. Default is "Train".
        subplot (int, optional): 
            Subplot position for plotting. Default is 121.
    """
    X, y = data
    
    # Predictions
    y_pred = model.predict(X)
    
    # Compute confusion matrix and evaluation metrics
    cm = confusion_matrix(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    # Create DataFrame for confusion matrix
    df_cm = pd.DataFrame(cm,
                         index=["Negative", "Positive"],
                         columns=["Predicted Negative", "Predicted Positive"])
    
    # Plot confusion matrix and Add title and evaluation metrics annotations
    plt.subplot(subplot)
    ax = plt.gca()
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 14})
    x = 0.5
    ax.text(1, -0.22, title, ha='center', fontsize=18)
    ax.text(x, -0.12, f"Precision={precision:.3f}", ha='center', fontsize=14)
    ax.text(x, -0.03, f"Recall={recall:.3f}", ha='center', fontsize=14)
    ax.text(x + 1, -0.12, f"Accuracy={accuracy:.3f}", ha='center', fontsize=14)
    ax.text(x + 1, -0.03, f"F1 Score={f1:.3f}", ha='center', fontsize=14)
    
    
def plot_confusion_matrices(model, data: tuple):
    """
    Plot confusion matrices for a classification model on both training and test data.

    Args:
        model: 
            The classification model.
        data (tuple): 
            Tuple containing X_train, X_test, y_train, and y_test.
    """
    X_train, X_test, y_train, y_test = data
    plt.figure(figsize=(16, 7))
    plot_confusion_matrix(model, (X_train, y_train), title="Train", subplot=121)
    plot_confusion_matrix(model, (X_test, y_test), title="Test", subplot=122)
    plt.tight_layout()
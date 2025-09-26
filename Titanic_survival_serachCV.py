import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# ======================
# Load Dataset
# ======================
df = pd.read_csv(r"C:\Users\saraye tel\OneDrive\Desktop\ARCH_Roadmap\Datasets\Titanic-Dataset.csv")

# Drop irrelevant columns
df.drop(columns=["PassengerId", "Name", "Ticket"], inplace=True, errors="ignore")

# ======================
# Handle Missing Values
# ======================
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df["Cabin"] = df["Cabin"].apply(lambda x: str(x)[0] if pd.notnull(x) else "U")  # keep only deck letter

# ======================
# Feature Engineering
# ======================
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df.drop(["SibSp", "Parch"], axis=1, inplace=True)

# ======================
# Encode Categorical
# ======================
categorical_cols = ["Sex", "Embarked", "Cabin"]
encoder = OneHotEncoder(drop="first", sparse_output=False)
encoded = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

df = pd.concat([df.drop(columns=categorical_cols).reset_index(drop=True),
                encoded_df.reset_index(drop=True)], axis=1)

# ======================
# Scale Numerical Features
# ======================
num_cols = ["Age", "Fare", "FamilySize"]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# ======================
# Split Data
# ======================
X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

# Handle imbalance with SMOTE
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# ======================
# Models
# ======================
models = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "MLP": MLPClassifier(max_iter=500)
}

accuracy_scores = {}

# ======================
# Training + Evaluation
# ======================
for name, model in models.items():
    
    # Hyperparameter tuning
    if name == "RandomForest":
        param_dist = {
            "n_estimators": [100, 200, 300],
            "max_features": ["sqrt"],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "bootstrap": [True]
        }
        rf_random = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                       n_iter=20, cv=3, random_state=42, n_jobs=-1)
        rf_random.fit(X_train, y_train)
        model = rf_random.best_estimator_

    elif name == "GradientBoosting":
        param_dist = {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
        gb_random = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                       n_iter=20, cv=3, random_state=42, n_jobs=-1)
        gb_random.fit(X_train, y_train)
        model = gb_random.best_estimator_

    elif name == "NeuralNetwork":
        param_dist = {
            "hidden_layer_sizes": [(50,), (100,), (50,50)],
            "activation": ["tanh", "relu"],
            "solver": ["adam"],
            "alpha": [0.0001, 0.001],
            "learning_rate": ["constant", "adaptive"]
        }
        nn_random = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                       n_iter=20, cv=3, random_state=42, n_jobs=-1)
        nn_random.fit(X_train, y_train)
        model = nn_random.best_estimator_
    
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    ras = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    accuracy_scores[name] = acc

    print("="*40)
    print(f"Model: {name}")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, pred))
    print("Confusion Matrix:\n", cm)
    print(f"ROC AUC Score: {ras:.4f}")

    # Confusion Matrix Heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'])
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {ras:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ======================
# Accuracy Comparison
# ======================
plt.figure(figsize=(8, 6))
sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()), palette="viridis")
plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
# 🚢 Titanic Survival Prediction

## 📌 Project Overview
This project applies **machine learning** to predict the survival of passengers aboard the Titanic, based on demographic and travel information.  
The dataset comes from the classic **Kaggle Titanic Dataset**, one of the most famous beginner ML challenges.  

The goal was to:
- Explore and preprocess the dataset  
- Engineer meaningful features  
- Train multiple machine learning models  
- Compare performance using accuracy, ROC-AUC, confusion matrices, and ROC curves  

---

## ⚙️ Steps in the Project
1. **Data Preprocessing**
   - Dropped irrelevant features: `PassengerId`, `Name`, `Ticket`  
   - Filled missing values in `Age` (median) and `Embarked` (mode)  
   - Simplified `Cabin` feature by keeping only the deck letter  

2. **Feature Engineering**
   - Created `FamilySize` = `SibSp + Parch + 1`  
   - Added `IsAlone` = 1 if passenger had no family, else 0  

3. **Encoding & Scaling**
   - One-Hot Encoding for categorical features (`Sex`, `Embarked`, `Cabin`)  
   - Standardized numerical features (`Age`, `Fare`, `FamilySize`)  

4. **Handling Class Imbalance**
   - Applied **SMOTE** (Synthetic Minority Oversampling) to balance survival classes  

5. **Model Training & Evaluation**
   - Trained 4 models:
     - Logistic Regression  
     - Random Forest  
     - Gradient Boosting  
     - Multi-Layer Perceptron (Neural Network)  
   - Evaluated using:
     - Accuracy  
     - Classification Report  
     - Confusion Matrix (with heatmaps)  
     - ROC-AUC Score  
     - ROC Curve plots  

---

## 📊 Results
- Logistic Regression: ~82% accuracy  
- Random Forest: ~87% accuracy  
- Gradient Boosting: ~89% accuracy  
- Neural Network: ~85% accuracy  

🔥 **Gradient Boosting performed the best with ~89% accuracy and strong ROC-AUC.**

---

## 📂 Repository Structure
├── Titanic-Dataset.csv # Dataset
├── titanic_survival.ipynb # Jupyter Notebook (optional)
├── titanic_survival.py # Main project script
└── README.md # Project documentation


---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/titanic-survival-prediction.git
pip install -r requirements.txt
python titanic_survival.py

## 🛠️ Technologies Used

Python
Pandas, NumPy
Scikit-learn
Imbalanced-learn (SMOTE)
Matplotlib, Seaborn

## ✨ Future Improvements

Apply hyperparameter tuning with RandomizedSearchCV / GridSearchCV
Try advanced models like XGBoost or LightGBM
Explore deeper feature engineering with passenger titles (Mr, Mrs, etc.)

## Author, Mahdiar

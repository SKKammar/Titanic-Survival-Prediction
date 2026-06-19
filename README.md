# 🚢 Titanic Survival Prediction

A complete end-to-end machine learning pipeline that benchmarks **six classification algorithms** on the Kaggle Titanic dataset. The project covers feature engineering, preprocessing, hyperparameter tuning with cross-validation, model comparison, and final model selection.

---

## 📂 Dataset

| Property | Value |
|---|---|
| Source | [Kaggle – Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data) |
| Training samples | 891 |
| Features | 12 raw (Age, Sex, Pclass, Fare, SibSp, Parch, etc.) |
| Target | `Survived` (0 = No, 1 = Yes) |

> Download `train.csv` from Kaggle and place it in the project root before running the notebook.

---

## ⚙️ Feature Engineering

Two features were engineered that provided the most significant performance gains across all models:

- **`Title`** — extracted from passenger names (Mr, Mrs, Miss, Master, Rare). Captures social class, age group, and marital status implicitly.
- **`FamilySize`** — computed as `SibSp + Parch + 1`. Both very large families and solo travellers showed lower survival rates.

**Columns dropped:** `PassengerId`, `Name`, `Ticket`, `Cabin`, `Embarked`  
(`Embarked` was excluded after showing near-zero feature importance in preliminary analysis.)

---

## 🧹 Preprocessing Pipeline

A unified `sklearn` pipeline was applied consistently across all six models before training:

| Feature Type | Imputation | Scaling / Encoding |
|---|---|---|
| Numeric (`Age`, `Fare`, `SibSp`, `Parch`, `FamilySize`) | Median | StandardScaler (zero mean, unit variance) |
| Categorical (`Sex`, `Title`) | Mode | One-Hot Encoding (drop first to avoid dummy trap) |

Data split: **80% train / 20% test** with stratification on the target variable.

---

## 🤖 Models & Hyperparameter Tuning

Each model was tuned using `GridSearchCV` with **5-fold cross-validation** on the training set.

| Model | Best Hyperparameters |
|---|---|
| Logistic Regression | `C = 10` |
| Random Forest | `n_estimators = 150`, `max_depth = 6`, `min_samples_split = 5` |
| KNN | `n_neighbors = 3` |
| Gradient Boosting | `learning_rate = 0.1`, `max_depth = 3`, `n_estimators = 100` |
| SVM (RBF kernel) | `C = 10`, `gamma = 'scale'` |
| MLP Neural Network | `hidden_layer_sizes = (50,)`, `alpha = 0.001`, `learning_rate_init = 0.01` |

---

## 📊 Results

Models ranked by test accuracy:

| Model | Test Accuracy | Precision | Recall | F1 Score | CV Accuracy (5-fold) |
|---|---|---|---|---|---|
| Gradient Boosting | **0.821** | 0.800 | 0.757 | 0.778 | 0.830 ± 0.021 |
| Random Forest | 0.816 | 0.797 | 0.743 | 0.769 | **0.834 ± 0.024** |
| Logistic Regression | 0.810 | 0.786 | 0.743 | 0.764 | 0.824 ± 0.015 |
| SVM (RBF) | 0.810 | 0.794 | 0.730 | 0.761 | 0.829 ± 0.014 |
| MLP Neural Network | 0.782 | 0.761 | 0.689 | 0.723 | 0.822 ± 0.025 |
| KNN | 0.732 | 0.691 | 0.635 | 0.662 | 0.806 ± 0.024 |

---

## 🏆 Best Model: Random Forest

**Random Forest** was selected as the final model based on the highest 5-fold CV accuracy (**83.4%**), which is the more reliable indicator of generalisation than a single test-set score. Gradient Boosting edges it out on raw test accuracy (82.1% vs 81.6%), but the difference is within the margin of variance, and Random Forest offers better interpretability via feature importances.

### Confusion Matrix — Random Forest

```
              Predicted: 0    Predicted: 1
Actual: 0         91              14
Actual: 1         19              55
```

- **True Negatives (correctly predicted not survived):** 91  
- **True Positives (correctly predicted survived):** 55  
- **False Positives:** 14  
- **False Negatives:** 19  

![Confusion Matrix](images/image1.png)

---

## 🧠 Key Takeaways

1. **Feature engineering was the biggest lever** — extracting `Title` alone produced the most consistent accuracy improvement across all six models.
2. **Gradient Boosting and Random Forest dominate** — their ability to capture non-linear interactions between features outpaces linear and distance-based models on this dataset.
3. **Logistic Regression is a strong baseline** — nearly matching ensemble methods at a fraction of the complexity, making it a good sanity-check benchmark.
4. **KNN and MLP underperformed** — KNN is sensitive to the feature space and scale; MLP requires careful regularisation and more data to generalise well.
5. **CV accuracy is the right selection criterion** — a single 80/20 split can be lucky or unlucky; 5-fold CV gives a more honest picture of model quality.

---

## 🚀 How to Reproduce

```bash
# 1. Clone the repository
git clone https://github.com/SKKammar/Titanic-Survival-Prediction.git
cd Titanic-Survival-Prediction

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# 3. Download train.csv from Kaggle and place it in the project root
# https://www.kaggle.com/c/titanic/data

# 4. Launch the notebook
jupyter notebook 3MLmodelProject.ipynb
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)

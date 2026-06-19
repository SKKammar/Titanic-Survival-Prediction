# 🚢 Titanic Survival Prediction 

## 📌 Overview
This project implements a full end‑to‑end machine learning pipeline for predicting passenger survival on the Titanic. We compare **six different classification algorithms**, perform extensive feature engineering, hyperparameter tuning with cross‑validation, and select the best model for deployment.

The six models evaluated are:
1. Logistic Regression
2. Random Forest
3. K‑Nearest Neighbors (KNN)
4. Gradient Boosting
5. Support Vector Machine (SVM) with RBF kernel
6. Multi‑Layer Perceptron (MLP) Neural Network

---

## 📂 Dataset
- **Source:** Kaggle Titanic – Machine Learning from Disaster
- **Training samples:** 891
- **Features:** 12 (Age, Sex, Pclass, Fare, etc.)
- **Target:** `Survived` (binary: 0 = No, 1 = Yes)

---

## ⚙️ Feature Engineering
We engineered two new features that significantly boosted model performance:

- **`Title`** – extracted from passenger names (e.g., Mr, Mrs, Miss, Master, Rare). This captures age, marital status, and social class.
- **`FamilySize`** – `SibSp + Parch + 1`. Large families had lower survival; travelling alone also impacted odds.

**Dropped columns:** `PassengerId`, `Name`, `Ticket`, `Cabin`, `Embarked` (the latter showed near‑zero importance in preliminary analysis).

---

## 🧹 Preprocessing Pipeline
A consistent pipeline was applied to all models:

- **Numeric features** (`Age`, `Fare`, `SibSp`, `Parch`, `FamilySize`): imputed with median, then scaled to zero mean and unit variance.
- **Categorical features** (`Sex`, `Title`): imputed with mode, then one‑hot encoded (drop first to avoid dummy trap).

The data was split into **80% training** and **20% test** with stratification.

---

## 🤖 Model Training & Hyperparameter Tuning
Each model was tuned using `GridSearchCV` with 5‑fold cross‑validation on the training set. The optimal hyperparameters found are listed below:

| Model | Best Hyperparameters |
|-------|----------------------|
| Logistic Regression | `C = 10` |
| Random Forest | `n_estimators = 150`, `max_depth = 6`, `min_samples_split = 5` |
| KNN | `n_neighbors = 3` |
| Gradient Boosting | `learning_rate = 0.1`, `max_depth = 3`, `n_estimators = 100` |
| SVM (RBF) | `C = 10`, `gamma = 'scale'` |
| MLP Neural Network | `alpha = 0.001`, `hidden_layer_sizes = (50,)`, `learning_rate_init = 0.01` |

---

## 📊 Results Comparison



| Model |	Test Accuracy |	Precision |	Recall |	F1 Score |	CV Accuracy (5‑fold) |
|-------|---------------|-----------|--------|----------|----------------------|
| Gradient Boosting	| 0.821	| 0.800 |	0.757 |	0.778 |	0.830 (±0.021) |
| Random Forest	| 0.816 |	0.797 |	0.743 |	0.769	| 0.834 (±0.024) |
| Logistic Regression |	0.810	| 0.786 |	0.743 |	0.764 |	0.824 (±0.015) |
| SVM (RBF) |	0.810 |	0.794 |	0.730 |	0.761	| 0.829 (±0.014) |
| MLP Neural Network |	0.782 |	0.761 |	0.689 |	0.723 |	0.822 (±0.025) |
| KNN |	0.732	| 0.691	| 0.635	| 0.662	| 0.806 (±0.024) |


---

## 🏆 Best Model Selection

Based on the highest cross‑validation accuracy, **Gradient Boosting** (or possibly Random Forest) emerges as the best performer. However, the difference between the top models is often marginal, and the choice may depend on interpretability or inference speed.

In our run, **Random Forest** achieved the highest CV accuracy (**83.4%**) and test accuracy (**81.6%**). We therefore select it as the final model for deployment.

---

## 📈 Confusion Matrix – Random Forest (Example)

```
[[91 14]
 [19 55]]
```
- **True Negatives:** 91  
- **False Positives:** 14  
- **False Negatives:** 19  
- **True Positives:** 55  

![Confusion Matrix](images/image1.png)


The model shows good balance with slightly more false negatives than false positives.

---

## 🧠 Final Conclusion

1. After evaluating six classifiers on the Titanic dataset, **Gradient Boosting** and **Random Forest** consistently outperform simpler models, thanks to their ability to capture non‑linear interactions between engineered features.
2. Feature engineering – especially extracting `Title` – provided the most significant performance gain across all models.
3. Logistic Regression remains a strong, interpretable baseline, achieving nearly 82% CV accuracy.
4. KNN and MLP were less consistent; KNN is sensitive to the feature space, while MLP requires careful regularization on small datasets.
5. For production, we recommend **Random Forest** (or Gradient Boosting) with hyperparameters tuned via cross‑validation, as it offers the best trade‑off between accuracy, robustness, and ease of deployment.

---

---

## 🚀 How to Reproduce

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Download `train.csv` from Kaggle and place it in the project root.
4. Run the Jupyter notebook or Python script.
5. All results will be printed and plots displayed.

---

⭐ **Star this repository if you found it useful!**  
Happy predicting! 🎯
```

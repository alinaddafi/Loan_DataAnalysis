# Loan_DataAnalysis
# 🧠 Loan Eligibility Prediction with Machine Learning

This project focuses on building and evaluating machine learning models to predict loan eligibility based on structured applicant data. The models were developed, tuned, and evaluated across various settings such as normalization, hyperparameter tuning, and feature selection.

## 📁 Dataset

The dataset contains anonymized information about loan applicants and their approval status. It includes both categorical and numerical features. The target variable is `Loan_Status`.

## 🔧 Project Workflow

The entire workflow was executed in the following stages:

1. **Data Cleaning & Preprocessing**  
   - Null values handled.
   - Categorical columns encoded using Label Encoding.
   - Data split into features (`X`) and target (`y`).
   - Dataset split into **train (65%)**, **validation (20%)**, and **test (15%)** sets.

2. **Normalization**  
   - Numeric features were normalized using `StandardScaler`.
   - Two versions of training and validation sets were prepared: with and without normalization.

3. **Model Training & Hyperparameter Tuning**  
   The following models were trained:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Artificial Neural Network (ANN using `MLPClassifier`)

   Each model was trained **with and without normalized data**, and evaluated on the validation set.

4. **Evaluation on Validation Set**  
   Accuracy scores of each model on validation data are reported below:

| Model                 | Without Scaling | With Scaling |
|----------------------|------------------|--------------|
| Logistic Regression  | 0.78             | 0.78         |
| KNN                  | 0.67             | 0.74         |
| ANN (MLPClassifier)  | 0.52             | **0.67**     |

5. **Final Evaluation on Test Set**  
   The final models were trained on the **full training data (train + validation)** and evaluated on the **test set** (scaled). Results are:

| Model                 | Accuracy |
|----------------------|----------|
| Logistic Regression  | **0.85** |
| KNN                  | 0.84     |
| ANN (MLPClassifier)  | 0.73     |

6. **Improvement Attempts**  
   Two strategies were attempted to improve performance:
   - **Feature Selection** using `SelectKBest`: Slight improvement in ANN accuracy, no gain in LR or KNN.
   - **Polynomial Features**: Led to **worse performance** in all models, likely due to overfitting and increased dimensionality.

## 📊 Visualization

A grouped bar chart was used to visualize the validation and test accuracies of all models (scaled vs unscaled where applicable).

## 🧠 Insights

- **Normalization** had a significant positive impact on **KNN** and **ANN**, but not on **Logistic Regression**.
- **ANN** benefited most from feature selection.
- **Polynomial features** added complexity but didn’t improve performance.

## 📝 Conclusion

While the baseline models achieved reasonably good performance, especially **Logistic Regression (0.85)** and **KNN (0.84)** on the test set, attempts to improve with advanced techniques yielded limited benefit. Future improvements could involve:
- Trying other models like **XGBoost**, **Random Forests**.
- Exploring more sophisticated feature engineering techniques.
- Performing cross-validation and ensembling.

---

📌 *Note: Despite applying two improvement techniques (feature selection and polynomial expansion), we did not observe meaningful performance gains across all models.*
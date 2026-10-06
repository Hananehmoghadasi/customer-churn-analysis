# Customer Churn Analysis

An end-to-end machine learning project for predicting customer churn and
translating model results into actionable retention insights.

## Project Overview

Customer churn is a major business challenge because retaining an
existing customer can be more valuable than acquiring a new one. This
project analyzes customer-level data to identify churn patterns, build
predictive models, compare their performance, and explain the final
model using SHAP.

The workflow covers:

-   Data cleaning and preprocessing
-   Exploratory data analysis (EDA)
-   Feature engineering
-   Baseline modeling
-   Model training and hyperparameter tuning
-   Model comparison
-   Nested out-of-fold threshold selection
-   Model evaluation
-   Feature importance and SHAP explainability
-   High-risk customer identification
-   Business recommendations

## Dataset

The project uses the **Telco Customer Churn** dataset.

Target variable:

-   `Churn`: whether the customer left the company (`Yes` / `No`)

The dataset contains customer demographics, account information,
contract type, payment method, internet services, and additional
services such as TechSupport and OnlineSecurity.

## Tools & Technologies

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   SHAP
-   Jupyter Notebook

## Methodology

### 1. Data Cleaning

Key preprocessing steps include:

-   Removing the customer identifier from modeling features
-   Converting `TotalCharges` to numeric format
-   Handling missing values
-   Encoding categorical variables
-   Preparing the target variable as a binary classification target

### 2. Feature Engineering

Additional features were created to improve the analysis, including:

-   `AvgMonthlyCharge`
-   `ServiceCount`
-   `IsNewCustomer`
-   `TenureGroup`

Categorical variables were transformed using one-hot encoding.

### 3. Models

Three classification approaches were compared:

1.  Logistic Regression
2.  Random Forest
3.  Gradient Boosting

Hyperparameters were optimized using `GridSearchCV` with ROC-AUC as the
scoring metric.

### 4. Threshold Selection

Instead of automatically using the default 0.50 classification
threshold, the threshold was selected using **nested out-of-fold
predictions** from the training data.

This prevents the test set from influencing threshold selection and
provides a more rigorous estimate of how the classification threshold
generalizes.

## Model Results

  -----------------------------------------------------------------------------
  Model          CV ROC-AUC Test ROC-AUC      Average   F1 (Churn)  Brier Score
                                            Precision              
  ------------ ------------ ------------ ------------ ------------ ------------
  Logistic           0.8484       0.8378       0.6415       0.6223       0.1721
  Regression                                                       

  **Gradient     **0.8477**   **0.8422**   **0.6593**   **0.6246**   **0.1372**
  Boosting**                                                       

  Random             0.8467       0.8379       0.6439       0.6154       0.1555
  Forest                                                           
  -----------------------------------------------------------------------------

### Final Model

**Gradient Boosting** was selected as the final model because it
achieved:

-   Test ROC-AUC: **0.8422**
-   Average Precision: **0.6593**
-   F1 Score: **0.6246**
-   Brier Score: **0.1372**
-   Nested OOF classification threshold: **0.1372**

The test set was kept untouched for final model evaluation.

## Model Explainability

Feature importance shows that the strongest predictive features include:

1.  `tenure`
2.  `InternetService_Fiber optic`
3.  `PaymentMethod_Electronic check`
4.  `Contract_Two year`
5.  `Contract_One year`
6.  `TotalCharges`
7.  `AvgMonthlyCharge`
8.  `MonthlyCharges`

### SHAP Findings

SHAP analysis provides both feature importance and direction of model
impact.

Key observations include:

-   **Tenure** is the most influential feature. Its SHAP pattern
    suggests a nonlinear and potentially interaction-dependent
    relationship with churn risk, so it should not be interpreted as a
    simple linear effect.
-   **One-year and two-year contracts** are generally associated with
    lower predicted churn risk than the reference contract category.
-   **Fiber optic customers** show higher predicted churn risk and
    represent an important segment for further investigation.
-   **Electronic-check customers** show higher predicted churn risk.
-   **Higher monthly charges** are associated with higher predicted
    churn risk.
-   **TechSupport** and **OnlineSecurity** are associated with lower
    predicted churn risk.

These are predictive associations identified by the model, not causal
conclusions.

## Business Insights & Recommendations

### 1. Prioritize month-to-month customers

Customers without long-term contracts represent an important retention
segment.

**Recommendation:** test contract-upgrade incentives and targeted
retention offers.

### 2. Investigate the Fiber optic segment

Fiber optic customers show higher predicted churn risk.

**Recommendation:** investigate whether pricing, service experience,
customer expectations, or other factors are associated with this pattern
before designing targeted interventions.

### 3. Monitor electronic-check customers

Electronic-check users show higher predicted churn risk.

**Recommendation:** analyze this customer segment further and consider
targeted retention campaigns.

### 4. Review high-charge customers

Higher monthly charges are associated with higher predicted churn risk.

**Recommendation:** investigate pricing/value perception and prioritize
high-risk customers for proactive engagement.

### 5. Consider service bundles

TechSupport and OnlineSecurity are associated with lower predicted churn
risk.

**Recommendation:** evaluate targeted service bundles for customers
identified as high risk.

### 6. Use the model for prioritization

The model can be used to rank customers by estimated churn risk and help
retention teams focus limited resources on the highest-priority
customers.

The classification threshold should be treated as a business decision as
well as a modeling parameter; the appropriate threshold can change
depending on the relative cost of false positives and false negatives.

## Limitations

-   The dataset is observational, so model relationships should not be
    interpreted as causal effects.
-   The selected threshold maximizes F1 on nested out-of-fold training
    predictions; a production system may require a threshold based on
    actual retention costs and capacity.
-   Customer behavior can change over time, so the model should be
    monitored and periodically re-evaluated with newer data.
-   SHAP explains model behavior; it does not prove that changing a
    feature will cause churn to increase or decrease.

## Project Structure

``` text
Customer-Churn-Analysis/
│
├── Customer_Churn_Analysis_clean_v4.ipynb
├── README.md
└── data/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Conclusion

This project demonstrates an end-to-end customer churn workflow that
combines predictive modeling with explainability and business analysis.

The final Gradient Boosting model achieved a **0.8422 Test ROC-AUC** and
**0.6593 Average Precision**, while SHAP analysis helped translate model
predictions into interpretable customer-risk patterns and practical
retention recommendations.

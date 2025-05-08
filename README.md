# Credit-Risk-Prediction-Using-Supervised-Machine-Learning-Model
A machine learning project for predicting credit card default risk using supervised machine learning techniques such as Logistic Regression, Random Forest, and XGBoost. Applied SMOTE and hyperparameter tuning for better performance. Developed for the ECON 611 term project at the University of Calgary.

## Contents
- `credit_risk_model.ipynb`: Full modelling Jupyter notebook.
- `Econ_611_Term_Report.pdf`: Final report detailing methodology, evaluation, and findings
- `README.md`: Project summary

## Objective
The goal of this project is to develop a predictive model that accurately classifies credit card clients into defaulters and non-defaulters. This will help lenders, such as banks 
and financial institutions, assess the likelihood that a specific borrower will default based on their personal and financial characteristics. 


## Dataset
- **Source**: [UCI Default of Credit Card Clients](https://doi.org/10.24432/C55S3H)
- **Size**: 30,000 records
- **Features**: 23 original attributes including demographics, billing, repayment status, and credit limit
- **Target Variable**: Default payment next month (1 =Client defaulted on payment, 0 = Client did not default)

## Methods & Techniques
### Preprocessing
- One-hot encoding for categorical variables
- StandardScaler for numeric features
- Train-test split (80/20)

### Feature Engineering
Derived features such as:
- Average credit utilization
- Number of late payments
- Recent delay risk
- Repayment-to-limit ratios

### Models Implemented
- Logistic Regression (with L1 & L2)
- Decision Tree
- Random Forest
- XGBoost
- K-Nearest Neighbours (KNN)
- Neural Network

### Class Imbalance Handling
- Used SMOTE to address class imbalance
- Applied only to the training data

### Hyperparameter Tuning
- Used GridSearchCV with 5-fold cross-validation
- Optimized Random Forest and Logistic Regression based on recall

## Evaluation Metrics
- Accuracy
- Recall (1)
- F1-Score
- AUC (1)
- Confusion Matrix

## Results Summary

| Model                 | Accuracy | Recall(1) | F1 Score(1)   | AUC |
|----------------------|--------|----------|-------|----------|
| Logistic Regression + SMOTE | 0.7360 | 0.62     | 0.51  | 74.0%    |
| Random Forest + SMOTE      | 0.6889   | 0.50  | 0.41 | 62%    |
| XGBoost + SMOTE            | 0.7797   | 0.48     | 0.49  | 74%      |
| KNN + SMOTE                | 0.8051   | 0.36     | 0.45 | 71%    |
| Neural Network + SMOTE         | 0.7814   | 0.50     | 0.50 | 78.14%    |

## Feature Importance (Top 3)

- `LONGEST_DELAY`: Max payment delay
- `PAY_0`: Most recent repayment status
- `NUM_DELAYS`: Total number of delayed months

## Key Takeaways
- **SMOTE** significantly improved recall without sacrificing too much AUC
- **Random Forest + SMOTE** gave the best balance of precision and sensitivity
- **Logistic Regression + SMOTE** had the highest recall, crucial for risk detection

## Business Implications
This pipeline provides a scalable and interpretable tool for financial institutions to:
- Minimize false negatives in risk assessment
- Improve loan portfolio management
- Deploy model in loan application systems & instantly flag high-risk clients for review.

## Author
**Bozhao Wang**  
Department of Economics  
University of Calgary  
Instructor: Dr. Arvind Magesan

## License
MIT License

---

**Note**: This project was submitted as a term report for ECON 611: Machine Learning in Economics. For academic use only.

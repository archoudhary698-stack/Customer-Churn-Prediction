# Customer Churn Prediction

A machine learning project where I predict whether a telecom customer will churn (leave the company) or stay, based on their account and usage details.

## Why this project?

Customer churn is a real problem for subscription-based businesses. Retaining an existing customer is cheaper than acquiring a new one, so if a company can predict which customers are likely to leave, it can offer them discounts or better plans in advance. I picked this dataset because it has a good mix of numerical and categorical features, which made it great for practising the full ML workflow.

## Dataset

- **Source:** Telco Customer Churn dataset (made available by IBM)
- The notebook loads the CSV directly from a public URL, so no manual download is needed.
- **Size:** 7,043 rows and 21 columns
- **Target column:** `Churn` (Yes / No)

## What I did

1. **Data cleaning** – Fixed the `TotalCharges` column (it was stored as text and had some blank values), filled missing values with the median, and dropped the `customerID` column.
2. **EDA** – Plotted the churn distribution, tenure vs churn, monthly charges vs churn, and churn by contract type. Around 26.5% of customers churn, so the data is somewhat imbalanced.
3. **Preprocessing** – Converted the target to 1/0, one-hot encoded the categorical columns using `pd.get_dummies`, and scaled the numerical columns with `StandardScaler`. I fit the scaler only on the training data to avoid data leakage.
4. **Model building** – Trained two models to compare:
   - Logistic Regression (baseline)
   - Random Forest
5. **Evaluation** – Compared the models using accuracy, confusion matrix, classification report and ROC-AUC. Since the classes are imbalanced, I focused more on recall and ROC-AUC than plain accuracy.
6. **Feature importance** – Used the Random Forest to find the biggest churn drivers.

## Results

| Model               | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| Logistic Regression | ~0.81    | ~0.84   |
| Random Forest       | ~0.79    | ~0.82   |

Logistic Regression performed slightly better and also works as a good interpretable baseline. The top churn factors were **tenure**, **monthly charges**, **total charges**, and **contract type** (month-to-month customers churn the most).

## Tech used

- Python
- pandas, NumPy – data handling
- Matplotlib, Seaborn – visualisation
- scikit-learn – preprocessing, models and evaluation

## How to run

The easiest way is Google Colab (nothing to install):

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook** and select `Customer_Churn_Prediction.ipynb`
3. Click **Runtime → Run all**

To run locally instead:

```bash
pip install -r requirements.txt
jupyter notebook Customer_Churn_Prediction.ipynb
```

## What I learned

- Why you should check column data types instead of trusting the CSV blindly (`TotalCharges` looked numeric but wasn't)
- What data leakage is and why the scaler must be fit only on training data
- Why accuracy alone is misleading on imbalanced data, and how ROC-AUC and recall give a better picture
- How to read a confusion matrix and a classification report
- Interpreting feature importance from a Random Forest

## Future improvements

- Hyperparameter tuning with GridSearchCV
- Handling class imbalance using SMOTE
- Trying XGBoost / Gradient Boosting

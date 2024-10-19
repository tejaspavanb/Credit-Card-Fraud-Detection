---

# Credit Card Fraud Detection

## Project Overview

Credit card fraud detection is a crucial application of machine learning that aims to identify fraudulent transactions in large datasets. 
In this project, I build a machine learning model to detect fraudulent credit card transactions using a highly imbalanced dataset from 
[Kaggle's Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).

This project covers:
- Data preprocessing and feature scaling.
- Handling class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)**.
- Building a **Logistic Regression** model.
- Model evaluation using **confusion matrix**, **classification report**, and **ROC-AUC score**.

---

## Dataset

The dataset used for this project is the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle. It contains transactions made by European cardholders over two days in September 2013. 

- **Transactions**: 284,807 total transactions.
- **Fraudulent Transactions**: 492 (only 0.172% of the dataset).
- **Features**: The dataset contains 30 features, including anonymized numerical features (`V1`, `V2`, ..., `V28`), `Amount`, `Time`, and `Class`.
  - **Class**: 0 for non-fraudulent, 1 for fraudulent transactions.

---

## Project Structure

```plaintext
├── credit_card_fraud_detection.ipynb  # Jupyter notebook with the full implementation
├── README.md                          # Project overview and details (this file)
└── dataset/                           # Folder to store the dataset
    └── creditcard.csv                 # The credit card fraud detection dataset
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tejaspavanb/Credit-Card-Fraud-Detection.git
   cd credit-card-fraud-detection
   ```

2. **Install the required dependencies**:
   Make sure you have Python 3.7 or higher. Install the required libraries by running:
   ```bash
   pip install -r requirements.txt
   ```
   The main libraries used in this project are:
   - `pandas`
   - `numpy`
   - `matplotlib`
   - `seaborn`
   - `scikit-learn`
   - `imbalanced-learn`

3. **Download the dataset** from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the `dataset/` directory.

4. **Run the notebook**:
   Open the Jupyter notebook and run the code to see the results.
   ```bash
   jupyter notebook credit_card_fraud_detection.ipynb
   ```

---

## Key Steps

### 1. Data Preprocessing

- **Feature Scaling**: The `Amount` column was scaled using **StandardScaler** to standardize the values.
- **Handling Class Imbalance**: The dataset is highly imbalanced, with only 0.172% of the transactions being fraudulent. To address this, I applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset by oversampling the minority class.

### 2. Model Training

I trained a **Logistic Regression** model on the processed data. The model was trained using both the original dataset and the oversampled dataset (after applying SMOTE).

### 3. Model Evaluation

The following metrics were used to evaluate the performance of the model:
- **Confusion Matrix**: To understand how many fraudulent and non-fraudulent transactions were correctly and incorrectly classified.
- **Classification Report**: To view precision, recall, and F1-score for each class (fraud/non-fraud).
- **ROC-AUC Curve**: To evaluate the trade-off between true positive rate and false positive rate.

---

## Results

- **Confusion Matrix**:
  The confusion matrix revealed that the model performs well in predicting non-fraudulent transactions, but it required handling of class imbalance to predict fraudulent transactions effectively.

- **Classification Report**:
  After applying SMOTE, the model achieved:
  - Precision: **0.85** for fraud class.
  - Recall: **0.82** for fraud class.
  - F1-Score: **0.83** for fraud class.

- **ROC-AUC Score**:
  The model achieved an **ROC-AUC score of 0.98**, demonstrating excellent ability to distinguish between fraud and non-fraud transactions.

- **ROC Curve**:
  ![ROC Curve](path_to_roc_curve_image.png)

---

## Technologies Used

- **Python**: Core programming language.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computing.
- **Matplotlib & Seaborn**: Data visualization.
- **scikit-learn**: Machine learning and evaluation tools.
- **imbalanced-learn**: Handling class imbalance with SMOTE.

---

## Future Work

- **Try more advanced models**: Random Forest, Gradient Boosting, or Neural Networks may offer better performance for this imbalanced dataset.
- **Hyperparameter Tuning**: Experiment with tuning the hyperparameters of the Logistic Regression model for further improvements.
- **Feature Engineering**: Analyze and derive more insights from the features, particularly the anonymized features (`V1`, `V2`, etc.).

---

## License

This project is licensed under the MIT License. See the [LICENSE](https://opensource.org/licenses/MIT) file for details.

---

## Acknowledgments

- Kaggle for providing the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).
- [imbalanced-learn](https://imbalanced-learn.org) for providing tools to handle imbalanced datasets.

---

### References

- [SMOTE: Synthetic Minority Over-sampling Technique](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [Logistic Regression - scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

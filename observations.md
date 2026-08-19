# Supervised ML Benchmark

## Dataset: Bank Marketing 

- **Total rows:** 11,162
- **Features:** 16
- **Target:** `deposit`
- **Problem type:** Binary Classification

The target represents whether a customer subscribed to a term deposit:

- `yes` → Customer subscribed
- `no` → Customer did not subscribe

### Feature Types

**Numerical Features:**

- `age`
- `balance`
- `day`
- `duration`
- `campaign`
- `pdays`
- `previous`

**Categorical Features:**

- `job`
- `marital`
- `education`
- `default`
- `housing`
- `loan`
- `contact`
- `month`
- `poutcome`

### Target Distribution

| Class | Percentage |
|---|---:|
| No | 52.62% |
| Yes | 47.38% |

So as you can see the classes are relatively balanced, but stratified splitting I still used to preserve the class distribution across the training, validation, and test datasets.

### Data Quality

Since **no missing values or duplicates were found**, no data was removed.

---

# Data Splitting

i split the dataset using a fixed random seed:

- **Training set:** 70%
- **Validation set:** 15%
- **Test set:** 15%
- **Random seed:** `42`

The validation set was used for model evaluation and comparison, while the final test set was kept separate for final evaluation.

---

# Preprocessing

I created preprocessing pipeline using `ColumnTransformer`.

## Numerical Features

Numerical features processed using:

1. `SimpleImputer(strategy="median")`
2. `StandardScaler()`

i selected median imputation because it is robust to potential outliers, and standard scaling because algorithms like **KNN and SVM are sensitive to feature magnitude and distance**.

## Categorical Features

for categorical features:

1. `SimpleImputer(strategy="most_frequent")`
2. `OneHotEncoder(handle_unknown="ignore")`

One-hot encoding converts categorical values into numerical features so machine learning algorithms can process. i also made sure to fit preprocessing pipeline only on training data to prevent any data leakage,

After preprocessing:

- **Training shape:** `(7813, 51)`
- **Validation shape:** `(1674, 51)`
- **Test shape:** `(1675, 51)`

---

# Models Implemented

I implemented all the models:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. Gradient Boosting
6. XGBoost
7. Support Vector Machine (SVM)

---

# Model Results

The following table shows the main selected version of each model.

| Model | Training Accuracy | Validation Accuracy | Train-Validation Gap | Inference Time (s) |
|---|---:|---:|---:|---:|
| Logistic Regression | 82.41% | 83.51% | - | **0.001063** |
| KNN (`K=15`) | 83.63% | 82.08% | - | 0.076378 |
| Decision Tree (`max_depth=8`) | 85.06% | 83.09% | 1.97% | 0.002633 |
| Random Forest | 100.00% | 86.44% | 13.56% | 0.065950 |
| Gradient Boosting | 85.58% | 85.78% | -0.21% | 0.014362 |
| XGBoost | 96.95% | 86.38% | 10.57% | 0.043171 |
| SVM with Scaling | 87.69% | **86.62%** | 1.07% | 2.814906 |

---

## Required Observations

1. **Which model performed best on training data?**
    Random Forest performed best on the training data with 100.00% training accuracy.

2. **Which model performed best on validation data?**
    SVM with Scaling performed best on the validation data with 86.62% validation accuracy

3. **Which model generalized best to the final test set?**
    SVM with Scaling generalized best to the final test set, achieving the highest performance on unseen test data. Its strong validation performance and small train-validation gap also indicate good generalization.

4. **Which algorithm was most interpretable?**
    Logistic Regression was the most interpretable because its coefficients show how each feature affects the prediction. Decision Tree was also highly interpretable due to its clear decision rules.

5. **Which algorithm was fastest at inference time?**
    Logistic Regression was the fastest algorithm at inference time, with a total inference time of 0.001063 seconds.

6. **Which model would you choose if explainability were a requirement?**
    Logistic Regression would be the preferred choice because its coefficients provide a direct and easily understandable explanation.

7. **Which model would you choose if predictive performance were the primary objective?**
    SVM with scaling would be the preferred model because it achieved the highest validation accuracy (86.62%) among all evaluated models. It also had a small train-validation gap (1.07%), indicating strong generalization and a good balance between predictive performance and overfitting.

8. **Did any model show signs of high bias or high variance?**
    Random Forest showed the strongest signs of high variance/overfitting, with 100.00% training accuracy but only 86.44% validation accuracy, resulting in a **13.56%** gap. XGBoost also showed signs of overfitting, with a 10.57% train-validation gap. In contrast, Gradient Boosting and SVM with Scaling showed relatively small gaps, suggesting better generalization. None of the models showed strong evidence of high bias based on these training and validation results.

## Effect of Class Imbalance

If dataset is imbalanced, so one class has significantly more examples than the other, it can affect model evaluation like this:

* A model may achieve high accuracy simply by predicting the majority class most of the time.
* The model may fail to correctly identify the less frequent class.
* The model tends to favor the majority class because it sees more examples of it during training.

In my Bank Marketing dataset, the classes were relatively balanced (~ **52.6% "no" and 47.4% "yes"**), so class imbalance didn't have a major effect on models' results.


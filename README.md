# Supervised ML Benchmark

## Setup and execution instructions

### Required Libraries

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost kagglehub
````

### Running the Project

1. Clone the repository:

```bash
git clone https://github.com/mashal-shakeel/ml_assignments.git
```

2. Navigate to the project directory:

```bash
cd ml_assignments
```

3. Open the notebooks using **Jupyter Notebook**:

```bash
jupyter notebook
```

4. Open and run the required model notebook, such as:

* `logistic_regression.ipynb`
* `knn.ipynb`
* `decision_tree.ipynb`
* `boosting_vs_random_forest.ipynb`
* `xg_boost.ipynb`
* `SVM.ipynb`

5. Run all cells in the notebook from top to bottom. The dataset is downloaded automatically using `kagglehub`.

### Google Colab

Alternatively, upload or open the notebooks in **Google Colab** and run all cells sequentially. Install any missing dependencies using:

```python
!pip install kagglehub xgboost
```


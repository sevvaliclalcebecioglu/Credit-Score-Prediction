# Credit Score Classification Project

This project is a machine learning application developed to predict individuals' credit scores. It encompasses steps such as data cleaning, model development, and creating an interactive interface.

---

## 📌 Project Steps

### 1. Data Cleaning
- I examined the data in a Jupyter Notebook and corrected missing/inconsistent values.
- The cleaned dataset (`clean_train.csv`) was saved and used for the next steps.

### 2. Model Development
- I tested 7 different classification models on the data using Python in VSCode:
  - GaussianNB
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - K-Nearest Neighbors (KNN)
  - AdaBoost
- Model performance was evaluated using **Accuracy, Precision, Recall, and F1 Score** metrics.

| Model             | Accuracy  | Precision | Recall   | F1       |
|------------------|-----------|-----------|---------|----------|
| RandomForest      | 0.777179  | 0.776636  | 0.777179| 0.776836 |
| GradientBoosting  | 0.698029  | 0.698980  | 0.698029| 0.697475 |
| DecisionTree      | 0.692177  | 0.692303  | 0.692177| 0.692234 |
| KNN               | 0.663433  | 0.658220  | 0.663433| 0.657464 |
| AdaBoost          | 0.639051  | 0.639703  | 0.639051| 0.639356 |
| GaussianNB        | 0.548198  | 0.549145  | 0.548198| 0.547254 |
| LogisticRegression| 0.541987  | 0.509543  | 0.541987| 0.476047 |

- The best-performing model was **Random Forest** and it was saved using `joblib` (`random_forest_model.pkl`).

### 3. Streamlit Interface
- An interactive **Streamlit interface** was created using the saved model, allowing users to input data.
- Users can enter information such as income, number of credit cards, and delay history to get a credit score prediction.

### 4. Live Demo
- The application was deployed on **Hugging Face Spaces**.
- Users can access the model via a browser and make predictions: [Credit Score Prediction App](https://huggingface.co/spaces/sevvaliclal/Credit-Score-Prediction-App)


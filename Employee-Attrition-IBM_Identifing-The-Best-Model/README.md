**Project: ML_Employee-Attrition-IBM_Identify-The-Best-Model**
**Purpose:**
The purpose of the above machine learning project is to predict employee attrition. Employee attrition refers to the phenomenon of employees leaving an organization voluntarily or involuntarily. Predicting attrition can be valuable for organizations as it allows them to identify factors and patterns that contribute to employee turnover. By understanding these factors, organizations can take proactive measures to improve employee satisfaction, engagement, and retention.

**Key Feature:**
This project is to develop and evaluate machine learning models that can accurately predict employee attrition. By identifying the factors that contribute to attrition and leveraging these predictive models, organizations can take proactive measures to reduce employee turnover, improve employee satisfaction, and maintain a stable workforce.

**Summary:**

1. Import the required libraries for data analysis and visualization, including NumPy, pandas, Matplotlib, and Seaborn.
2. Read the dataset using `pd.read_csv()` function and set the index of the DataFrame to "EmployeeNumber" using `data.set_index()`.
3. Check for missing values in the dataset using `data.isnull().sum()`.
4. Check for duplicated rows in the dataset using `data.duplicated().sum()`.
5. Separate the features (X) and the target variable (y) by dropping the "Attrition" column from the DataFrame.
6. Perform exploratory data analysis by visualizing the data using histograms and heatmaps.
7. Identify the columns with object data type and store them in the `col_obj` list.
8. Analyze the distribution of numerical features using boxplots.
9. Use Seaborn to create count plots to visualize the relationship between the target variable ("Attrition") and categorical variables such as "JobRole", "Department", "EducationField", "Gender", "MaritalStatus", and "OverTime".
10. Calculate the skewness of numerical features using `data.skew()` and identify features with skewness greater than 0.9.
11. Apply a square root transformation to the skewed features to reduce skewness.
12. Encode categorical variables using LabelEncoder from sklearn.preprocessing.
13. Split the data into training and testing sets using `train_test_split()` from sklearn.model_selection.
14. Create a list of machine learning models to be evaluated, including Logistic Regression, Random Forest, SVM, KNN, Decision Tree, and Gaussian NB.
15. Perform model evaluation using cross-validation and calculate accuracy and ROC AUC scores.
16. Store the evaluation results in a DataFrame and sort them based on the ROC AUC mean score.
17. Visualize the accuracy results using boxplots.

Please note that the code provided assumes that the necessary libraries and the dataset file are already available and properly imported.

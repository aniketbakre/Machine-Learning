# Titanic Spaceship Classification Project

In this project, I performed classification tasks on the Titanic Spaceship dataset. The dataset consists of various features related to passengers on the Titanic and whether they were transported or not. The goal was to build a machine learning model to predict the transport status based on the given features.

## Data Preprocessing

I started by importing the necessary libraries, including pandas and numpy, and loaded the train and test datasets. Then, I performed some initial data exploration and preprocessing steps on the train dataset:

1. Checked for missing values using `isnull().sum().sort_values(ascending=False)` and dropped rows with missing values using `dropna()`.
2. Set the "PassengerId" column as the index using `set_index("PassengerId")`.
3. Removed irrelevant columns ("HomePlanet", "Cabin", "Destination", "Name") from the dataset using `drop()`.

For categorical columns, I used LabelEncoder from scikit-learn to convert them into numerical values.

## Correlation Analysis

To check the correlation between the features and the target variable ("Transported"), I calculated the correlation matrix using `corr()`. Then, I extracted the correlation values of each feature with the target variable and sorted them in descending order. This helped identify the most correlated features.

Based on the correlation analysis, I dropped some features that had low correlation with the target variable ("VIP", "Age", "VRDeck", "Spa", "RoomService").

## Model Building and Evaluation

Next, I trained several classification models using scikit-learn. The models used were:

1. Logistic Regression
2. Random Forest
3. SVM (Support Vector Machines)
4. K-Nearest Neighbors
5. Decision Tree
6. Gaussian Naive Bayes

For each model, I performed cross-validation using KFold with 10 splits and evaluated their performance using accuracy and ROC AUC score.

The results were stored in a DataFrame, including the algorithm name, ROC AUC mean and standard deviation, accuracy mean and standard deviation. The results were sorted based on the ROC AUC mean.

## Visualization

To visualize the performance of the algorithms, I created a bar plot using matplotlib. The plot showed the ROC AUC mean and accuracy mean for each algorithm. I also added value labels on top of each bar to display the corresponding scores.

Based on the plot, it was evident that Gaussian Naive Bayes had the highest accuracy and ROC AUC mean, indicating its superior performance among the algorithms tested.

## Final Evaluation

To validate the performance of the Gaussian Naive Bayes model, I trained it on the entire training dataset (`x_train`, `y_train`) and made predictions on the test dataset (`x_test`). I then calculated the accuracy score using the predicted labels and the actual test labels (`y_test`).

The final accuracy score indicated the overall performance of the model on the unseen data.

Overall, this project involved data preprocessing, feature selection, model training, evaluation, and visualization to classify the transport status of passengers on the Titanic Spaceship. The Gaussian Naive Bayes algorithm demonstrated the best performance for this dataset.

Note: Make sure to update the file paths for the dataset in the code to match your specific file locations.

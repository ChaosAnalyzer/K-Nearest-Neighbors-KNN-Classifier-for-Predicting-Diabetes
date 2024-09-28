 ##### Project Title - K-Nearest Neighbors (KNN) Classifier for Predicting Diabetes

### Project Overview:
This study applies the K-Nearest Neighbours (KNN) classifier to the Pima Indians Diabetes Database in order to predict the onset of diabetes. The dataset consists of one target variable (Outcome) that indicates if an individual has diabetes (1) or not (0), together with a number of medical predictor variables.


### Data Description:
Dataset: Pima Indians Diabetes Database
Source: [https://www.kaggle.com/datasets/mathchi/diabetes-data-set]
Features:
Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age
Target: Outcome (1: Diabetes, 0: Healthy)

Exploratory Data Analysis (EDA):
Libraries Used:
pandas
numpy
seaborn
matplotlib

Summary Statistics:

Displayed dataset statistics such as mean, median, min, max values.
Found no missing values in the dataset.

Data Visualizations:

Histograms for each feature.
Pair plots to analyze the relationship between features, with Outcome as the hue.
KDE plots to understand the distribution of features for diabetic and non-diabetic patients.
Heatmap for the correlation between features.
Boxplot to visualize feature distribution and outliers.

### Data Preprocessing:
Scaling: Used StandardScaler to normalize the feature set.
Train-Test Split: Split the dataset into training and testing sets with an 80-20 ratio.

 code
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


### K-Nearest Neighbors (KNN) Classifier:
Model Initialization: Initialized the KNN classifier with n_neighbors=5.
Model Training: Trained the model on the training dataset.
code
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
Prediction: Predicted outcomes for the test set.
 code
y_pred = knn.predict(X_test)
Evaluation:
Accuracy: 69%
Precision, Recall, F1-score for both classes (Healthy, Diabetic) were reported.
 code
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
Confusion Matrix & Classification Report:
Precision for detecting healthy individuals was higher than for diabetic patients, indicating potential room for improvement.

### K-Value Optimization:
Cross-Validation: Tested different values of K (from 1 to 30) using cross-validation.
Best K: The optimal value of K was found by plotting accuracy for each K value.
 code
from sklearn.model_selection import cross_val_score
k_range = range(1, 31)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    scores.append(score.mean())

plt.plot(k_range, scores)
plt.xlabel('Value of K')
plt.ylabel('Cross-Validated Accuracy')
plt.title('K-Value Optimization for Model Accuracy')
plt.show()


# Conclusion:
Findings: The KNN classifier achieved an accuracy of around 69%. The model performed better at predicting non-diabetic outcomes compared to diabetic outcomes, indicating the need for potential improvements or a more complex model.
Future Work: Exploring other classifiers like Random Forest, SVM, or improving the feature engineering might enhance the model's performance.

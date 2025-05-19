#diabetes prediction
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#Load the dataset
df = pd.read_csv(r"C:\diabetes.csv")

# Display the first rows of the dataset
print("first rows of the data set:",df.head)
print("shape of the dataset is:",df.shape)

#Display the columns names of the data set
print("columns names of the data set:",df.columns)
# Display the data types of the columns
print("data types of the columns:",df.dtypes)
# Display the summary statistics of the dataset
print("summary statistics of the dataset:",df.describe())
# Display the number of missing values in each column
print("number of missing values in each column:",df.isnull().sum())
# Display the distribution of the target variable
print("distribution of the target variable:",df['Outcome'].value_counts())

#Display the correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm')
plt.title("correlation_matrix")
plt.show()
#Train Test and split 
x = df.drop('Outcome', axis = 1)
y = df['Outcome']
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=42)
#scale the features
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# Train the model
# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(x_train,y_train)
# Predict the test set results
y_pred=log_reg.predict(x_test)
# Evaluate the model
print("Accuracy of the model:",accuracy_score(y_test,y_pred))
print("Confusion matrix of the model:",confusion_matrix(y_test,y_pred))
print("Classification report of the model:",classification_report(y_test,y_pred))
print("ROC AUC score of the model:",roc_auc_score(y_test,y_pred))
# Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
# Predict the test set results
y_pred_rf=rf.predict(x_test)
# Evaluate the model
print("Accuracy of the model:",accuracy_score(y_test,y_pred_rf))
print("Confusion matrix of the model:",confusion_matrix(y_test,y_pred_rf))
print("Classification report of the model:",classification_report(y_test,y_pred_rf))
print("ROC AUC score of the model:",roc_auc_score(y_test,y_pred_rf))
# SVM Classifier
svm = SVC()
svm.fit(x_train,y_train)
# Predict the test set results
y_pred_svm=svm.predict(x_test)
# Evaluate the model
print("Accuracy of the model:",accuracy_score(y_test,y_pred_svm))
print("Confusion matrix of the model:",confusion_matrix(y_test,y_pred_svm))
print("Classification report of the model:",classification_report(y_test,y_pred_svm))
print("ROC AUC score of the model:",roc_auc_score(y_test,y_pred_svm))

#feauture importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
for f in range(x.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(x.shape[1]), importances[indices], align="center")
    plt.xticks(range(x.shape[1]), indices)
    plt.xlim([-1, x.shape[1]])
    plt.show()
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(x_train, y_train)
    print("Best parameters from grid search:", grid_search.best_params_)
    # Train the model with the best parameters
    rf_best = RandomForestClassifier(**grid_search.best_params_)
    rf_best.fit(x_train, y_train)
    # Predict the test set results
    y_pred_rf_best = rf_best.predict(x_test)
    # Evaluate the model
    print("Accuracy of the model:", accuracy_score(y_test, y_pred_rf_best))
    print("Confusion matrix of the model:", confusion_matrix(y_test, y_pred_rf_best))
    print("Classification report of the model:", classification_report(y_test, y_pred_rf_best))
    print("ROC AUC score of the model:", roc_auc_score(y_test, y_pred_rf_best))
    # Decision Tree Classifier
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    # Predict the test set results
    y_pred_dt = dt.predict(x_test)
    # Evaluate the model
    print("Accuracy of the model:", accuracy_score(y_test, y_pred_dt))
        
    print("Confusion matrix of the model:", confusion_matrix(y_test, y_pred_dt))
    print("Classification report of the model:", classification_report(y_test, y_pred_dt))
    print("ROC AUC score of the model:", roc_auc_score(y_test, y_pred_dt))
    # KNN Classifier
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    # Predict the test set results
    y_pred_knn = knn.predict(x_test)
    # Evaluate the model
    print("Accuracy of the model:", accuracy_score(y_test, y_pred_knn))
    print("Confusion matrix of the model:", confusion_matrix(y_test, y_pred_knn))
    print("Classification report of the model:", classification_report(y_test, y_pred_knn))
    print("ROC AUC score of the model:", roc_auc_score(y_test, y_pred_knn))
    # Naive Bayes Classifier
    nb = GaussianNB()
    nb.fit(x_train, y_train)
    # Predict the test set results
    y_pred_nb = nb.predict(x_test)
    # Evaluate the model
    print("Accuracy of the model:", accuracy_score(y_test, y_pred_nb))
    print("Confusion matrix of the model:", confusion_matrix(y_test, y_pred_nb))
    print("Classification report of the model:", classification_report(y_test, y_pred_nb))
    print("ROC AUC score of the model:", roc_auc_score(y_test, y_pred_nb))
    # Plot ROC curve
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    # Calculate the false positive rate and true positive rate
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_rf_best)
    # Calculate the area under the curve
    roc_auc = auc(fpr, tpr)
    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()
# Save the model
import joblib
joblib.dump(rf_best, 'diabetes_model.pkl')
# Load the model
rf_loaded = joblib.load('diabetes_model.pkl')
     
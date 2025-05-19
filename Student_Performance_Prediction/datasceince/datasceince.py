import pandas as pd
import os
import matplotlib.pyplot as plt 
import seaborn as sns 
f = os.path.join(r"C:\Users\linga\Desktop\data\student-mat.csv")
df = pd.read_csv(f, sep=';')
print(df.head())
print(df.info())
print(df.describe())
print(df.columns)
print(df.index)
print(df.values)
print(df.shape)
print(df.size)
print(df.ndim)
print(df.dtypes)
print(df.isnull().sum())
# One-hot encode categorical columns
df_encoded = pd.get_dummies(df, drop_first=True)

# Now compute correlation
correlation_matrix = df_encoded.corr()
g3_corr = correlation_matrix['G3'].sort_values(ascending=False)

print("Top correlations with G3:")
print(g3_corr.head(10))


# correlation matrix
print("Correlation of features with G3:")
print(g3_corr)
plt.figure(figsize=(10, 8))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm',fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
#normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import Normalizer
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Read the data (with sep=';')
df = pd.read_csv(r"C:\Users\linga\Desktop\data\student-mat.csv", sep=';')

# Step 2: Create binary target variable
df['passfail'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
print(df['passfail'].value_counts())

# Step 3: Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Step 4: Split features and target
X = df_encoded.drop(['G3', 'passfail'], axis=1)
y = df_encoded['passfail']

# Step 5: Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Train a logistic regression model
model = LogisticRegression(max_iter=1000)  # Increase max_iter to avoid convergence warnings
model.fit(X_train, y_train)

# Step 8: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('Support Vector Machine', SVC())
]


# Train and evaluate each model
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\n🔹 Model: {name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    import joblib

# Assume random forest is the best
best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)

# Save it
joblib.dump(best_model, 'student_pass_predictor_model.pkl')
print("Model saved as student_pass_predictor_model.pkl")

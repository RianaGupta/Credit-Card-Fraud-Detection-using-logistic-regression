import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


df=pd.read_csv("creditcard.csv")
print(df.head())

# ----- CLASS=0 (genuine)-------CLASS=1(fraud)
print(df["Class"].value_counts())

#------ independent and dependent functions------
X = df.drop("Class", axis=1)
y = df["Class"]

# ----Train,Test split------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ----SCALING---
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------TRAINING MODEL----
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --PREDICITON-----
y_pred = model.predict(X_test)
# ---
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
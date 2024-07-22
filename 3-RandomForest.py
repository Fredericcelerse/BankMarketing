#!/usr/bin/env python

# First we need to load the data
import pandas as pd
import random
import numpy as np

df = pd.read_csv('./Databases/bank.csv', delimiter=';')
df.head()

# Set the random seeds for reproducibility
random.seed(123)
np.random.seed(123)

# We sparse the data
yes_data = df[df['y'] == "yes"]
no_data = df[df['y'] == "no"]

# We preprocess again these two tables as we did before, and we standardize
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def preprocess_data(X):
    X = X.copy()
    label_encoder = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].fillna('missing')
        X[col] = label_encoder.fit_transform(X[col])
    for col in X.select_dtypes(include=['float64', 'int64']).columns:
        X[col] = X[col].fillna(X[col].median())
    return X

yes_data.iloc[:,-1] = label_encoder.fit_transform(yes_data.iloc[:,-1])
yes_data.iloc[:,-1] = 1 - yes_data.iloc[:,-1]
no_data.iloc[:,-1] = label_encoder.fit_transform(no_data.iloc[:,-1])

yes_data.iloc[:,:-1] = preprocess_data(yes_data.iloc[:,:-1])
no_data.iloc[:,:-1] = preprocess_data(no_data.iloc[:,:-1])

yes_data.iloc[:,:-1] = scaler.fit_transform(yes_data.iloc[:,:-1])
no_data.iloc[:,:-1] = scaler.fit_transform(no_data.iloc[:,:-1])

# We perform PCA analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
yes_pca = pca.fit_transform(yes_data)
pca_no = PCA(n_components=2)
no_data_pca = pca_no.fit_transform(no_data)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(no_data_pca[:, 0], no_data_pca[:, 1], alpha=0.5)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA on the yes_data')
plt.grid(True)
#plt.show()
#plt.savefig("PCA.png", dpi=600)

# We extract the most diverse "no data"
n_yes = yes_data.shape[0]
from sklearn.metrics import pairwise_distances
distances = pairwise_distances(yes_pca, no_data_pca)
min_distances_indices = distances.min(axis=0).argsort()[:n_yes]
diverse_no_data = no_data.iloc[min_distances_indices]

remaining_no_data = no_data.drop(diverse_no_data.index)

diverse_no_data.to_csv("Databases/diverse_no_data.csv", index=False)
remaining_no_data.to_csv("Databases/remaining_no_data.csv", index=False)

# We create new features
combined_data = pd.concat([yes_data, diverse_no_data])

X_features_new = combined_data.iloc[:,:-1]
y_target_new = combined_data.iloc[:,-1]

from sklearn.preprocessing import PolynomialFeatures
def create_features(X, combination, feature_names):
    poly = PolynomialFeatures(degree=combination, interaction_only=False, include_bias=False)
    X_poly = poly.fit_transform(X)
    return pd.DataFrame(X_poly, columns=poly.get_feature_names_out(feature_names))

from scipy.stats import pearsonr
def calculate_correlations(X, y):
    correlations = {}
    for col in X.columns:
        corr, _ = pearsonr(X[col], y)
        correlations[col] = abs(corr)
    return correlations

X_features_enhanced = create_features(X_features_new, 3, X_features_new.columns)
correlations = calculate_correlations(X_features_enhanced, y_target_new)
print("Correlations with the target -> combination = 3:")
new_sorted_correlations = sorted(correlations.items(), key=lambda item: item[1], reverse=True)
print(new_sorted_correlations[:5])  # Print only the 5 best correlations

new_best_features = [feature for feature, corr in new_sorted_correlations[:5]]

# We finally build and train different Random Forest models
# We make 10 tests of splitting of the selected database and we train each time a RF with standard hyperparameters and we report the result
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

models_RF = []
for i in range(10):
    print("Training {}:".format(i+1))
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_features_enhanced[new_best_features], y_target_new, test_size=0.2)
    if y_train_new.dtype == 'object':
        le = LabelEncoder()
        y_train_new = le.fit_transform(y_train_new)
    if y_test_new.dtype == 'object':
        le = LabelEncoder()
        y_test_new = le.fit_transform(y_test_new)
    model_RF = RandomForestClassifier()
    model_RF.fit(X_train_new, y_train_new)
    y_pred_new = model_RF.predict(X_test_new)
    print(f'Optimized Accuracy: {accuracy_score(y_test_new, y_pred_new)}')
    models_RF.append(model_RF)

# We now test the 10 different models on the remaining data

X_remaining = remaining_no_data.drop('y', axis=1)
y_remaining = remaining_no_data['y']
if y_remaining.dtype == 'object':
    le = LabelEncoder()
    y_remaining = le.fit_transform(y_remaining)

X_remaining_new = create_features(X_remaining, 3, X_remaining.columns)
X = X_remaining_new[new_best_features]
y = y_remaining
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

for i in range(10):
    print("Test {}".format(i+1))
    y_remaining_pred = models_RF[i].predict(X)
    remaining_accuracy = accuracy_score(y, y_remaining_pred)
    print(f'Accuracy on remaining no_data: {remaining_accuracy}')

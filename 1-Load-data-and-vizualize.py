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

# Then we need to define the features and the data, and the clean the data 
X_features = df.iloc[:, :-1]
y_target = df.iloc[:,-1]

# We need to first visualize the data in order to see what we have

numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

import matplotlib.pyplot as plt
import seaborn as sns

# We create histograms for numerical columns
for col in numeric_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
    #plt.savefig(f'Histogram of {col}', dpi=600)

# and circular diagrams for string data
for col in categorical_columns:
    plt.figure(figsize=(10, 6))
    df[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=140)
    plt.title(f'Circular diagram of {col}')
    plt.ylabel('')  
    plt.show()
    #plt.savefig(f'Circular diagram of {col}', dpi=600)

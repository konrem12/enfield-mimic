#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import zipfile
import gzip
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
import networkx as nx
import matplotlib.pyplot as plt
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
from sklearn.neural_network import MLPRegressor
import dowhy
from dowhy import CausalModel
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz
from causallearn.utils.GraphUtils import GraphUtils
from econml.dml import CausalForestDML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder


####From here downwards, we read all the available files found
### in both hosp and icu folders

## Some of the files was not readable due to their size



# Define the file path
zip_path = "your path"

def read_csv_from_zip(zip_path, gz_path_within_zip):
    try:
        # Open the zip file
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Extract the .gz file to memory
            with z.open(gz_path_within_zip) as gz_file:
                # Open the .gz file
                with gzip.open(gz_file, 'rt') as csv_file:
                    # Load the CSV data into a pandas DataFrame
                    df = pd.read_csv(csv_file, encoding='utf-16')
                    return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# If the file is too large, we can alternativelly use this code and create batches


def read_largecsv_zip(zip_path, csv_inside_zip,chunksize):
    # Ask user for the chunk number
    selected_chunk = int(input("Enter the chunk number to load (starting from 0): "))

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(csv_inside_zip) as gz_file:
            with gzip.open(gz_file, "rt") as csv_file:
                # Read CSV in chunks
                for chunk_num, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunksize)):
                    if chunk_num == selected_chunk:
                        print(f"Loaded chunk {chunk_num}:")
                        print(chunk.head())  # Display first few rows
                        break  # Stop after finding the required chunk
                return chunk


.....


data.to_csv('icu_dataset.csv', index=False)

# add new variables; one counting how many time someone have been admitted to icu and another to work as a flag for readmission cases. 

def add_icu_stay_count_and_readmission(df, patient_col='subject_id', stay_col='stay_id', time_col='intime'):
    df_copy = df.copy()
    
    # Ensure datetime for ordering
    df_copy[time_col] = pd.to_datetime(df_copy[time_col])
    
    # Sort by patient and time
    df_copy = df_copy.sort_values([patient_col, time_col])
    
    # Readmission flag: 0 for first ICU stay, 1 for subsequent
    df_copy['readmission'] = df_copy.groupby(patient_col).cumcount().apply(lambda x: 1 if x > 0 else 0)
    
    # ICU stay count: total number of ICU stays per patient
    df_copy['icu_stay_count'] = df_copy.groupby(patient_col)[stay_col].transform('count')
    
    return df_copy

# we took also the timestamps of the admissions and created new variables showing the time of the day, the day of the week and 
# the month within the year the readmissions where made.

def process_intime(df, time_col='intime'):
    df_copy = df.copy()

    # Convert to datetime
    df_copy[time_col] = pd.to_datetime(df_copy[time_col])

    # Extract features
    df_copy[time_col + '_hour'] = df_copy[time_col].dt.hour
    df_copy[time_col + '_dayofweek'] = df_copy[time_col].dt.dayofweek
    df_copy[time_col + '_month'] = df_copy[time_col].dt.month

    # Drop the original timestamp
    df_copy = df_copy.drop(columns=[time_col])

    return df_copy

# finally we created a encoding for the categorical variables

def encode_categorical(df):
    """
    Automatically encodes categorical variables using Label Encoding.

    Args:
        df: pandas DataFrame

    Returns:
        df_encoded: pandas DataFrame with categorical variables encoded
        encoders: dictionary of fitted LabelEncoders for inverse_transform if needed
    """
    df_encoded = df.copy()
    encoders = {}

    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

        # Optional: treat columns with small number of unique values as categorical too
        elif df[col].nunique() < 10 and df[col].dtype in [np.int64, np.float64]:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col])
            encoders[col] = le

    return df_encoded, encoders

# DAG graph creation


# Create networkx graph from causal-learn output
G_nx = nx.DiGraph()




# Run PC algorithm (with correlation-based test as default)

cg = pc(data, labels=labels)



# cg.G is the learned graph (adjacency matrix)
adj_mat = cg.G.graph

for i in range(len(labels)):
    for j in range(len(labels)):
        if adj_mat[i, j] != 0:  # 0: no edge, 1: i → j, -1: j → i, 2: undirected
            if adj_mat[i, j] == 1:
                G_nx.add_edge(labels[i], labels[j])
            elif adj_mat[i, j] == -1:
                G_nx.add_edge(labels[j], labels[i])
            elif adj_mat[i, j] == 2:
                G_nx.add_edge(labels[i], labels[j])
                G_nx.add_edge(labels[j], labels[i])

# Plot with matplotlib
color_map = []
for node in G_nx.nodes():
    if node == 'readmission':
        color_map.append('red')  # Target variable highlighted in red
    else:
        color_map.append('lightblue')
plt.figure(figsize=(10,7))
pos = nx.spring_layout(G_nx, k=0.5, seed=42)
nx.draw(G_nx, pos, with_labels=True, node_color=color_map, node_size=3000, arrowsize=20)
plt.title("Learned DAG from PC Algorithm")
plt.show()





# Run GES algorithm
result = ges(data)


# In[65]:


# Access adjacency matrix correctly
adj = result['G'].graph

# Build networkx graph
G_nx = nx.DiGraph()

for i in range(len(labels)):
    for j in range(len(labels)):
        if adj[i, j] != 0:
            G_nx.add_edge(labels[i], labels[j])

# Highlight 'readmit'
color_map = ['red' if node == 'readmission' else 'lightblue' for node in G_nx.nodes()]

# Plot the DAG
plt.figure(figsize=(10,7))
pos = nx.spring_layout(G_nx, seed=42)
nx.draw(G_nx, pos, with_labels=True, node_color=color_map, node_size=3000, arrows=True)

plt.title("GES Learned DAG")
plt.show()




#Get FCI algorithm

cg, sep_set = fci(data, fisherz, alpha=0.05)



# Build networkx graph
G_nx = nx.DiGraph()
pag = cg.graph


# In[68]:


for i in range(len(labels)):
    for j in range(len(labels)):
        if pag[i, j] != 0:
            G_nx.add_edge(labels[i], labels[j], label=str(pag[i, j]))

            
color_map = []
for node in G_nx.nodes():
    if node == 'readmission':
        color_map.append('red')  # Target variable highlighted in red
    else:
        color_map.append('lightblue')

        
plt.figure(figsize=(15,15))
pos = nx.spring_layout(G_nx, seed=42)

nx.draw(G_nx, pos, with_labels=True, node_color=color_map, node_size=3000, arrows=True)

edge_labels = nx.get_edge_attributes(G_nx, 'label')
nx.draw_networkx_edge_labels(G_nx, pos, edge_labels=edge_labels)

plt.title("FCI Learned DAG")
plt.show()


# In[ ]:




# From now on we find the causal effects through MPL model  


X = data.drop('readmission', axis =1).to_numpy().astype('float32')  # Covariates
T1 = data['target_variable'].to_numpy().ravel().astype('float32')       # Treatment
Y = data['readmission'].to_numpy().ravel().astype('float32')   # Outcome


# In[70]:


mlp_model = MLPRegressor(hidden_layer_sizes=(64, 32),
                         activation='relu',
                         solver='adam',
                         max_iter=500,
                         random_state=42)

model = CausalForestDML(model_y=mlp_model,
                        model_t=mlp_model,
                        n_estimators=500,
                        random_state=42,   # <-- Control forest randomness
                        min_samples_leaf=10,
                        max_depth=10)

model.fit(Y, T1, X=X)

ate = model.ate(X)   # Average Treatment Effect over the population
print("Estimated ATE:", ate)
# For the average treatment effect (ATE)
ate_interval = model.ate_interval(X)
print("ATE Confidence Interval:", ate_interval)



# Now to train both traditional and vausal model, based on the HistGradientBoostingClassifier



X_reg = X.copy()
causal_parents = ['los', 'value', 'itemid', 'icu_stay_count', 'ordercategoryname']

for col in X.columns:
    if col not in causal_parents:
        X_reg[col] *= 0.01  # Downweight non-causal features

aucsum = 0

for i in range(10): 
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y, test_size=0.3, stratify=y)
    # Train model
    clf = HistGradientBoostingClassifier(max_iter=500)
    clf.fit(X_train, y_train)

    aucsum += roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])


# Evaluate
print(classification_report(y_test, clf.predict(X_test)))
print("ROC AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
print("the mean of ROC AUC for 10 iter:", aucsum/(i+1))


# In[249]:


from sklearn.ensemble import HistGradientBoostingClassifier
# Define features and target
X = clean_dataset_encoded.drop('readmission', axis =1)
y = clean_dataset_encoded['readmission']              # ICU readmission (binary outcome)

aucsum = 0

for i in range(10): 
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    clf = HistGradientBoostingClassifier(max_iter=500)
    clf.fit(X_train, y_train)

    aucsum += roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

print(classification_report(y_test, clf.predict(X_test)))
print("ROC AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
print("the mean of ROC AUC for 10 iter:", aucsum/(i+1))


# Finally, run the SHAP for explainability

# Use TreeExplainer (native support for HistGradientBoostingClassifier)
explainer = shap.Explainer(clf, X_test)

# Compute SHAP values
shap_values = explainer(X_test, check_additivity=False)

# Summary Plot (Global Importance)
plt.figure(figsize=(10,6))
plt.savefig('shap.png')
shap.summary_plot(shap_values, X_test)



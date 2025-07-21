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


# In[2]:


from sklearn.preprocessing import LabelEncoder


# In[3]:


###From here downwards, we read all the available files found
### in both hosp and icu folders

## Some of the files was not readable due to their size


# In[4]:


# Define the file path
zip_path = r"D:\\mimic-iv-31.zip"

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


# In[5]:


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


# In[ ]:





# In[6]:


icustays = read_csv_from_zip("D:/mimic-iv-31.zip", "mimic-iv-3.1/icu/icustays.csv.gz")


# In[7]:


patients = read_csv_from_zip("D:/mimic-iv-31.zip", "mimic-iv-3.1/hosp/patients.csv.gz")


# In[8]:


procedureevents = read_csv_from_zip("D:/mimic-iv-31.zip"
                                    , "mimic-iv-3.1/icu/procedureevents.csv.gz")


# In[9]:


helper_1 = patients.merge(icustays, on= ["subject_id"]).drop('anchor_year_group', axis = 1)


# In[45]:


helper_2 = helper_1.merge(procedureevents, on = ['subject_id', 'hadm_id', 'stay_id'] ).drop(['statusdescription', 
                                                                                             'continueinnextdept', 
                                                                                             'caregiver_id', 
                                                                                            'location', 
                                                                                             'locationcategory', 
                                                                                             'linkorderid', 
                                                                                            'storetime', 'endtime', 
                                                                                             'last_careunit', 'dod',
                                                                                            'outtime', 'originalrate',
                                                                                            'ordercategorydescription'], axis=1)


# In[46]:


helper_2.columns


# In[47]:


dataset = helper_2.dropna()


# In[48]:


dataset.to_csv('icu_dataset.csv', index=False)


# In[49]:


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


# In[50]:


dataset_icucounted = add_icu_stay_count_and_readmission(dataset)


# In[51]:


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


# In[52]:


dataset_timed = process_intime(dataset_icucounted)


# In[55]:


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


# In[56]:


dataset_encoded, encoders = encode_categorical(dataset_timed)


# In[57]:


encoders


# In[58]:


encoders['gender'].transform(["M", "F"])


# In[59]:


clean_dataset_encoded  =  dataset_encoded


# In[60]:


clean_dataset_encoded.columns


# In[61]:


# Create networkx graph from causal-learn output
G_nx = nx.DiGraph()


# In[62]:


# Run PC algorithm (with correlation-based test as default)
columns = clean_dataset_encoded.drop(['stay_id'], axis = 1).columns
data_array = clean_dataset_encoded.drop(['stay_id'], axis = 1).to_numpy()
cg = pc(data_array, labels=clean_dataset_encoded.columns)


# In[63]:


# cg.G is the learned graph (adjacency matrix)
adj_mat = cg.G.graph

for i in range(len(columns)):
    for j in range(len(columns)):
        if adj_mat[i, j] != 0:  # 0: no edge, 1: i → j, -1: j → i, 2: undirected
            if adj_mat[i, j] == 1:
                G_nx.add_edge(columns[i], columns[j])
            elif adj_mat[i, j] == -1:
                G_nx.add_edge(columns[j], columns[i])
            elif adj_mat[i, j] == 2:
                G_nx.add_edge(columns[i], columns[j])
                G_nx.add_edge(columns[j], columns[i])

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
plt.savefig('DAG_PC.png')
plt.show()


# In[64]:


# Run GES
result = ges(data_array)


# In[65]:


# Access adjacency matrix correctly
adj = result['G'].graph

# Build networkx graph
G_nx = nx.DiGraph()

for i in range(len(columns)):
    for j in range(len(columns)):
        if adj[i, j] != 0:
            G_nx.add_edge(columns[i], columns[j])

# Highlight 'readmit'
color_map = ['red' if node == 'readmission' else 'lightblue' for node in G_nx.nodes()]

# Plot the DAG
plt.figure(figsize=(10,7))
pos = nx.spring_layout(G_nx, seed=42)
nx.draw(G_nx, pos, with_labels=True, node_color=color_map, node_size=3000, arrows=True)

plt.title("GES Learned DAG")
plt.show()


# In[ ]:





# In[66]:


cg, sep_set = fci(data_array, fisherz, alpha=0.05)


# In[67]:


# Build networkx graph
G_nx = nx.DiGraph()
pag = cg.graph


# In[68]:


for i in range(len(columns)):
    for j in range(len(columns)):
        if pag[i, j] != 0:
            G_nx.add_edge(columns[i], columns[j], label=str(pag[i, j]))

            
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
plt.savefig('DAG_FCI.png')
plt.show()


# In[ ]:





# In[ ]:


var_names = columns  # Your variable names in correct order
idx_map = {var: i for i, var in enumerate(var_names)}

adj_matrix = np.zeros((len(var_names), len(var_names)))

for src, tgt in G_nx.edges():
    adj_matrix[idx_map[src], idx_map[tgt]] = 1


# In[69]:


X = clean_dataset_encoded.drop('readmission', axis =1).to_numpy().astype('float32')  # Covariates
T1 = clean_dataset_encoded['los'].to_numpy().ravel().astype('float32')       # Treatment
Y = clean_dataset_encoded['readmission'].to_numpy().ravel().astype('float32')   # Outcome


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


# In[ ]:





# In[71]:


T2 = clean_dataset_encoded['gender'].to_numpy().ravel().astype('float32')       # Treatment


# In[72]:


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

model.fit(Y, T2, X=X)

ate = model.ate(X)   # Average Treatment Effect over the population
print("Estimated ATE:", ate)
# For the average treatment effect (ATE)
ate_interval = model.ate_interval(X)
print("ATE Confidence Interval:", ate_interval)


# In[ ]:





# In[79]:


T3 = clean_dataset_encoded['icu_stay_count'].to_numpy().ravel().astype('float32')       # Treatment


# In[80]:


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

model.fit(Y, T3, X=X)

ate = model.ate(X)   # Average Treatment Effect over the population
print("Estimated ATE:", ate)
# For the average treatment effect (ATE)
ate_interval = model.ate_interval(X)
print("ATE Confidence Interval:", ate_interval)


# In[ ]:





# In[81]:


T4 = clean_dataset_encoded['intime_month'].to_numpy().ravel().astype('float32')       # Treatment


# In[82]:


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

model.fit(Y, T4, X=X)

ate = model.ate(X)   # Average Treatment Effect over the population
print("Estimated ATE:", ate)
# For the average treatment effect (ATE)
ate_interval = model.ate_interval(X)
print("ATE Confidence Interval:", ate_interval)


# In[ ]:





# In[77]:


T5 = clean_dataset_encoded['anchor_age'].to_numpy().ravel().astype('float32')       # Treatment


# In[78]:


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

model.fit(Y, T4, X=X)

ate = model.ate(X)   # Average Treatment Effect over the population
print("Estimated ATE:", ate)
# For the average treatment effect (ATE)
ate_interval = model.ate_interval(X)
print("ATE Confidence Interval:", ate_interval)


# In[ ]:





# In[84]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report


# In[184]:


# Define features and target

X = clean_dataset_encoded.drop('readmission', axis =1)
y = clean_dataset_encoded['readmission']              # ICU readmission (binary outcome)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = Y)


# In[199]:


# Train model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))


# In[248]:


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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[228]:


# Use TreeExplainer (native support for HistGradientBoostingClassifier)
explainer = shap.Explainer(clf, X_test)

# Compute SHAP values
shap_values = explainer(X_test, check_additivity=False)

# Summary Plot (Global Importance)
plt.figure(figsize=(10,6))
plt.savefig('shap.png')
shap.summary_plot(shap_values, X_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[101]:


import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


# In[103]:


features = [col for col in clean_dataset_encoded.columns if col not in ['readmission']]
target = 'readmission'


# In[114]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(clean_dataset_encoded[features], clean_dataset_encoded[target], test_size=0.3, random_state=42)

# DMatrix without feature weights (no regularization)
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

# Define parameters
params = {
    'objective': 'binary:logistic',
    'max_depth': 5,
    'eta': 0.05,
    'lambda': 1.0,
    'eval_metric': 'logloss'
}

# Train model (no regularization)
model = xgb.train(params, dtrain, num_boost_round=200)

# Predict
y_pred_prob = model.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluate
print("Accuracy (No Regularization):", accuracy_score(y_test, y_pred))
print("AUC (No Regularization):", roc_auc_score(y_test, y_pred_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature importance
plt.figure(figsize=(10,8))
xgb.plot_importance(model, max_num_features=15)
plt.title('Feature Importance (XGBoost without Causal Regularization)')
plt.show()

# SHAP explainability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10,8))
shap.summary_plot(shap_values, X_test, feature_names=features)


# In[ ]:





# In[ ]:





# In[116]:


causal_parents = ['icu_stay_count', 'item_id', 'los', 'value', 'first_careunit']


# In[117]:


X_train, X_test, y_train, y_test = train_test_split(clean_dataset_encoded[features], clean_dataset_encoded[target], test_size=0.3, random_state=42)


# In[130]:


feature_weights = []
for col in features:
    if col in causal_parents:
        feature_weights.append(1.0)  # no penalty
    else:
        feature_weights.append(5.0)  # penalize non-causal features


# In[131]:


# Convert to DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train, feature_weights=np.array(feature_weights), feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)


params = {
    'objective': 'binary:logistic',
    'max_depth': 5,
    'eta': 0.05,
    'lambda': 1.0,   # L2 regularization (optional)
    'eval_metric': 'logloss'
}

model = xgb.train(params, dtrain, num_boost_round=200)


# In[134]:


# Predict
y_pred_prob = model.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)


# In[135]:


# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_pred_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[110]:


# Plot feature importance
plt.figure(figsize=(10,8))
xgb.plot_importance(model, max_num_features=15)
plt.title('Feature Importance (Causal Regularized XGBoost)')
plt.show()


# In[112]:


import shap


# In[113]:


# SHAP explainability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10,8))
shap.summary_plot(shap_values, X_test, feature_names=features)


# In[141]:


from tqdm import tqdm
from sklearn.utils import resample

# Parameters
n_bootstrap = 100
top_n = 10  # Number of top features to consider in stability

# Initialize importance storage
importance_regularized = pd.DataFrame(0.0, index=features, columns=range(n_bootstrap))
importance_standard = pd.DataFrame(0.0, index=features, columns=range(n_bootstrap))

# Bootstrap loop
for i in tqdm(range(n_bootstrap)):
    # Bootstrap sample
    df_sample = resample(clean_dataset_encoded, replace=True, n_samples=len(clean_dataset_encoded), random_state=i)

    X_sample = df_sample[features]
    y_sample = df_sample[target]

    # Standard XGBoost
    dtrain_std = xgb.DMatrix(X_sample, label=y_sample, feature_names=features)

    params = {
        'objective': 'binary:logistic',
        'max_depth': 5,
        'eta': 0.05,
        'lambda': 1.0,
        'eval_metric': 'logloss'
    }

    model_std = xgb.train(params, dtrain_std, num_boost_round=100)

    std_importance = model_std.get_score(importance_type='gain')
    for feat in std_importance:
        importance_standard.loc[feat, i] = std_importance[feat]

    # Causal Regularized XGBoost
    feature_weights = [1.0 if col in causal_parents else 5.0 for col in features]
    dtrain_causal = xgb.DMatrix(X_sample, label=y_sample, feature_weights=np.array(feature_weights), feature_names=features)

    model_causal = xgb.train(params, dtrain_causal, num_boost_round=100)

    causal_importance = model_causal.get_score(importance_type='gain')
    for feat in causal_importance:
        importance_regularized.loc[feat, i] = causal_importance[feat]

# Compute selection frequency in top N
def compute_selection_frequency(importance_df):
    freq = pd.Series(0, index=importance_df.index)
    for i in range(n_bootstrap):
        top_features = importance_df.iloc[:,i].sort_values(ascending=False).head(top_n).index
        freq[top_features] += 1
    return freq / n_bootstrap

stability_std = compute_selection_frequency(importance_standard)
stability_causal = compute_selection_frequency(importance_regularized)

# Plot stability comparison
plt.figure(figsize=(10,6))
plt.scatter(stability_std, stability_causal, c=['red' if f in causal_parents else 'blue' for f in features], s=100)
plt.xlabel('Standard XGBoost Stability (Selection Frequency)')
plt.ylabel('Causal Regularized XGBoost Stability')
plt.title(f'Bootstrap Feature Stability (Top {top_n} Features)')
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

# Annotate points
for feat in features:
    plt.annotate(feat, (stability_std[feat], stability_causal[feat]), fontsize=8)

plt.show()


# In[145]:


# Predict
y_pred_prob = model_causal.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)


# In[146]:


# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_pred_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[ ]:





from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,f1_score, cohen_kappa_score
from sklearn import preprocessing
import pandas as pd
import numpy as np
from util.classifier import classifier

df=pd.read_csv('features.csv')


## Training and testing data NOTE: random state to keep the test always the same
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

## Split into features and cancer values
x=df.drop(columns='Cancer')
y=df['Cancer']

## Split based on cancer values
for train_idx, test_idx in sss.split(x, y):
    train_df = df.iloc[train_idx]  # 80%
    test_df = df.iloc[test_idx]    # 20%
## Gives training data (80%) and testing data (20%)

## Normalize the test features
#Separate cancer values
test_df_cancer=test_df.copy()
test_df=test_df.drop(columns='Cancer')
columns = test_df.columns
#Scaler get
scaler = preprocessing.StandardScaler().fit(test_df)
#Scaler apply
test_scaled=scaler.transform(test_df)
#Transform back to DF
test_df_scaled=pd.DataFrame(test_scaled,columns=columns,dtype=np.float64)

rfc_accuracies=[]
rfc_recalls=[]
rfc_f1=[]
best_model=None

## Cross validation
for _ in range(10):
    ## Training and validation
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    train_df2=train_df.copy()
    ## Split into features and cancer values
    x = train_df2.drop(columns='Cancer')
    y = train_df2['Cancer']

    ## Split based on cancer values
    for train_idx, test_idx in sss.split(x, y):
        final_train_df = train_df.iloc[train_idx]  # 80%
        valid_df = train_df.iloc[test_idx]    # 20%
    ## Takes previous training data
    ## Gives training data (80%) and validation data (20%)

    ## Remove 'Cancer' for normalizing features
    train_df_cancer=final_train_df.copy()
    train_df2=final_train_df.drop(columns='Cancer')

    valid_df_cancer=valid_df.copy()
    valid_df=valid_df.drop(columns='Cancer')
    
    ## Scaling
    scaler = preprocessing.StandardScaler().fit(train_df2)
    train_scaled = scaler.transform(train_df2)
    valid_scaled = scaler.transform(valid_df)
    train_df_scaled = pd.DataFrame(train_scaled,columns=columns,dtype=np.float64)
    valid_df_scaled= pd.DataFrame(valid_scaled,columns=columns,dtype=np.float64)

    knntrained,rfctrained,gpctrained=classifier(train_df_scaled, train_df_cancer)
    predicted_rfc=rfctrained.predict(valid_df_scaled)

    if rfc_recalls:
        if recall_score(valid_df_cancer['Cancer'],predicted_rfc) > max(rfc_recalls):
            best_model=rfctrained
    else:
        best_model=rfctrained

    rfc_accuracies.append(accuracy_score(valid_df_cancer['Cancer'],predicted_rfc))
    rfc_recalls.append(recall_score(valid_df_cancer['Cancer'],predicted_rfc))
    rfc_f1.append(f1_score(valid_df_cancer['Cancer'],predicted_rfc))

print(f'Average of RFC accuracies over 10 runs: {np.mean(rfc_accuracies)}')
print(f'Average of RFC recalls over 10 runs: {np.mean(rfc_recalls)}')
print(f'Average of RFC F1 scores over 10 runs: {np.mean(rfc_f1)}')
print('Test part')
## Testing
#predicted_knn_test=knntrained.predict(test_df_scaled)
predicted_rfc_test=best_model.predict(test_df_scaled)
#predicted_gpc_test=gpctrained.predict(test_df_scaled)

#acc_knn = accuracy_score(test_df_cancer['Cancer'], predicted_knn_test)
acc_rfc = accuracy_score(test_df_cancer['Cancer'], predicted_rfc_test)
#acc_gpc = accuracy_score(test_df_cancer['Cancer'], predicted_gpc_test)
#print('knn -',acc_knn)
print('rfc -',acc_rfc)
#print('gpc -',acc_gpc)

#recall_knn = recall_score(test_df_cancer['Cancer'],predicted_knn_test)
recall_rfc = recall_score(test_df_cancer['Cancer'],predicted_rfc_test)
#recall_gpc = recall_score(test_df_cancer['Cancer'],predicted_gpc_test)
#print('knn recall -',recall_knn)
print('rfc recall -',recall_rfc)
#print('gpc recall -',recall_gpc)

#roc_knn=roc_auc_score(test_df_cancer['Cancer'],predicted_knn_test)
roc_rfc=roc_auc_score(test_df_cancer['Cancer'],predicted_rfc_test)
#roc_gpc=roc_auc_score(test_df_cancer['Cancer'],predicted_gpc_test)
#print('knn roc -',roc_knn)
print('rfc roc -',roc_rfc)
#print('gpc roc -',roc_gpc)

#f1_knn=f1_score(test_df_cancer['Cancer'],predicted_knn_test)
f1_rfc=f1_score(test_df_cancer['Cancer'],predicted_rfc_test)
#f1_gpc=f1_score(test_df_cancer['Cancer'],predicted_gpc_test)
#print('knn f1 -',f1_knn)
print('rfc f1 -',f1_rfc)
#print('gpc f1 -',f1_gpc)

#cohen_kappa_knn = cohen_kappa_score(test_df_cancer['Cancer'], predicted_knn_test)
cohen_kappa_rfc= cohen_kappa_score(test_df_cancer['Cancer'], predicted_rfc_test)
#cohen_kappa_gpc= cohen_kappa_score(test_df_cancer['Cancer'], predicted_gpc_test)
#print('knn cohen kappa -', cohen_kappa_knn)
print('rfc cohen kappa -', cohen_kappa_rfc)
#print('gpc cohen kappa -', cohen_kappa_gpc)

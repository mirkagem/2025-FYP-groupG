import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,f1_score, cohen_kappa_score
from sklearn import preprocessing
sys.path.append(os.path.abspath("util"))

#Importing the loader class, features and classifier.
import util.img_util as loadI
from util.feature_A import feature_A
from util.feature_B import feature_B
from util.feature_C import feature_C
from util.feature_H import feature_H
from util.classifier import classifier

#This function is used to find the path where the data is kept, i.e. the picture and masks for the picture(if there are any)
#It's done this way so we avoid hard coded paths and so it runs without any user modifications
def path_finder():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    relative_path_to_data = os.path.join(current_directory, './data')
    data_folder_path = os.path.normpath(relative_path_to_data)
    return data_folder_path

#Loader class initialization
loaderclass = loadI.ImageDataLoader(path_finder())

#List of images, masks and image names
imageBatch = loaderclass.__iter__()

#Initialize DataFrame for features
df = pd.DataFrame()

imgIDx = 0

#Code for finding the path of the CSVs
#It's done this way so we avoid hard coded paths and so it runs without any user modifications
metadata_dir = os.path.dirname(os.path.abspath(__file__)) + '/' + 'metadata.csv'
annotation_dir = os.path.dirname(os.path.abspath(__file__)) +'/result' + '/' + 'manual_annotations.csv'

#Loading the CSVs
df_ground=pd.read_csv(metadata_dir)

df_ground['cancer']=np.where( (df_ground[ 'diagnostic']=='BCC') ^ (df_ground['diagnostic']=='MEL') ^ (df_ground['diagnostic']=='SCC'),1,0)

df_annotations= pd.read_csv(annotation_dir)

#We use this for to apply the features to the picture and to get a dataframe which includes all of the data returned from them
for imgAndMask in imageBatch:
    img = imgAndMask[0]
    mask = imgAndMask[1]
    patNumber = imgAndMask[2]

    hairy=df_annotations['Average'][(df_annotations['img_id']==patNumber)].to_numpy()
    cancer_part=df_ground['cancer'][( df_ground['img_id']==patNumber )].to_numpy()

    feature_A(img,mask,imgIDx,df)
    feature_B(img,mask,imgIDx,df)
    feature_C(img,mask,imgIDx,df)
    feature_H(img, imgIDx, df)

    df.loc[imgIDx,'Cancer']=cancer_part[0]

    imgIDx += 1

#New DF with values from Feature_H and averages of annotations
df_new=pd.DataFrame()
df_new['manual'] = df_annotations['Average']
df_new['model'] = df['Hair']

#Trinarize values from Feature_H - 1%=0, 10%=1, >10%=2 FIXME: Use this somewhere, it's not part of the classifier (and should not be)
def zero_two(x):
    if x<0.01:
        return 0
    elif x<0.1:
        return 1
    else:
        return 2  
df_new['transformed']= df_new['model'].apply(zero_two)

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

## Setup for cross-validation
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
    
    ## Scaling the training and validation data
    scaler = preprocessing.StandardScaler().fit(train_df2)
    train_scaled = scaler.transform(train_df2)
    valid_scaled = scaler.transform(valid_df)
    train_df_scaled = pd.DataFrame(train_scaled,columns=columns,dtype=np.float64)
    valid_df_scaled= pd.DataFrame(valid_scaled,columns=columns,dtype=np.float64)

    ## Train the model
    knntrained,rfctrained,gpctrained=classifier(train_df_scaled, train_df_cancer)
    ## Predict with validation data
    predicted_rfc=rfctrained.predict(valid_df_scaled)

    ## Save best performing model
    if rfc_recalls:
        if recall_score(valid_df_cancer['Cancer'],predicted_rfc) > max(rfc_recalls):
            best_model=rfctrained
    else:
        best_model=rfctrained

    ## Collect accuracy metrics
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

#acc_knn = accuracy_score(test_df_cancer['Cancer'], predicted_knn_test)
acc_rfc = accuracy_score(test_df_cancer['Cancer'], predicted_rfc_test)
#print('knn -',acc_knn)
print('rfc -',acc_rfc)

#recall_knn = recall_score(test_df_cancer['Cancer'],predicted_knn_test)
recall_rfc = recall_score(test_df_cancer['Cancer'],predicted_rfc_test)
#print('knn recall -',recall_knn)
print('rfc recall -',recall_rfc)

#roc_knn=roc_auc_score(test_df_cancer['Cancer'],predicted_knn_test)
roc_rfc=roc_auc_score(test_df_cancer['Cancer'],predicted_rfc_test)
#print('knn roc -',roc_knn)
print('rfc roc -',roc_rfc)

#f1_knn=f1_score(test_df_cancer['Cancer'],predicted_knn_test)
f1_rfc=f1_score(test_df_cancer['Cancer'],predicted_rfc_test)
#print('knn f1 -',f1_knn)
print('rfc f1 -',f1_rfc)

#cohen_kappa_knn = cohen_kappa_score(test_df_cancer['Cancer'], predicted_knn_test)
cohen_kappa_rfc= cohen_kappa_score(test_df_cancer['Cancer'], predicted_rfc_test)
#print('knn cohen kappa -', cohen_kappa_knn)
print('rfc cohen kappa -', cohen_kappa_rfc)

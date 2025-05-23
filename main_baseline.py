import os
import sys
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath("util"))

import util.img_util as loadI
from util.feature_A import feature_A
from util.feature_B import feature_B
from util.feature_C import feature_C

def path_finder():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    relative_path_to_data = os.path.join(current_directory, './data')
    data_folder_path = os.path.normpath(relative_path_to_data)
    return data_folder_path

test = loadI.ImageDataLoader(path_finder())

testBatch = test.__iter__()

df = pd.DataFrame()

imgIDx = 0

for imgAndMask in testBatch:
    img = imgAndMask[0]
    mask = imgAndMask[1]

    feature_A(mask,imgIDx,df)
    feature_B(img, mask, imgIDx, df)
    feature_C(img, imgIDx, df)

    imgIDx += 1

csv_dir = os.path.dirname(os.path.abspath(__file__)) + '/' + 'metadata.csv'

df_ground=pd.read_csv(csv_dir)
df_ground['cancer']=np.where( (df_ground[ 'diagnostic']=='BCC') ^ (df_ground['diagnostic']=='MEL') ^ (df_ground['diagnostic']=='SCC'),1,0)

df['Cancer']=df_ground['cancer']

## Training and testing data NOTE: stratify with equal amount of men and women
from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(5,shuffle=True)

## Split into features and cancer values
X=df.drop(columns='Cancer')
y=df['Cancer']

## Split based on cancer values
for train_idx, test_idx in skf.split(X, y):
    train_df = df.iloc[train_idx]  # 80%
    test_df = df.iloc[test_idx]    # 20%
    break
## Gives training data (80%) and testing data (20%)

## Training and validation
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4)

## Split into features and cancer values
X = train_df.drop(columns='Cancer')
y = train_df['Cancer']

## Split based on cancer values
for train_idx, test_idx in sss.split(X, y):
    train_df = df.iloc[train_idx]  # 60%
    valid_df = df.iloc[test_idx]    # 40%
## Takes previous training data
## Gives training data (60%) and validation data (40%)

## Remove 'Cancer' for normalizing features
train_df_cancer=train_df.copy()
train_df=train_df.drop(columns='Cancer')

valid_df_cancer=valid_df.copy()
valid_df=valid_df.drop(columns='Cancer')
## Normalize the features
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(train_df)

#Apply to data to both train and test test
train_scaled = scaler.transform(train_df)
valid_scaled = scaler.transform(valid_df)

train_df_scaled = pd.DataFrame(train_scaled,columns=['A - asymmetry','B - convexity', 'C - blueveil'],dtype=np.float64)
valid_df_scaled= pd.DataFrame(valid_scaled,columns=['A - asymmetry','B - convexity', 'C - blueveil'],dtype=np.float64)

## Primitive way of finding accuracy
## Train a classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knntrained = knn.fit(train_df_scaled, train_df_cancer['Cancer']) ## Training data and cancer values

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfctrained=rfc.fit(train_df_scaled,train_df_cancer['Cancer'])

from sklearn.gaussian_process import GaussianProcessClassifier
gpc=GaussianProcessClassifier()
gpctrained=gpc.fit(train_df_scaled,train_df_cancer['Cancer'])

## TEMP until we find a better way of finding accuracy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,roc_auc_score,f1_score
predicted_knn=knntrained.predict(valid_df_scaled)
predicted_rfc=rfctrained.predict(valid_df_scaled)
predicted_gpc=gpctrained.predict(valid_df_scaled)
acc_knn = accuracy_score(valid_df_cancer['Cancer'], predicted_knn)
acc_rfc = accuracy_score(valid_df_cancer['Cancer'], predicted_rfc)
acc_gpc = accuracy_score(valid_df_cancer['Cancer'], predicted_gpc)
print('knn -',acc_knn)
print('rfc -',acc_rfc)
print('gpc -',acc_gpc)

recall_knn = recall_score(valid_df_cancer['Cancer'],predicted_knn)
recall_rfc = recall_score(valid_df_cancer['Cancer'],predicted_rfc)
recall_gpc = recall_score(valid_df_cancer['Cancer'],predicted_gpc)
print('knn recall -',recall_knn)
print('rfc recall -',recall_rfc)
print('gpc recall -',recall_gpc)

roc_knn=roc_auc_score(valid_df_cancer['Cancer'],predicted_knn)
roc_rfc=roc_auc_score(valid_df_cancer['Cancer'],predicted_rfc)
roc_gpc=roc_auc_score(valid_df_cancer['Cancer'],predicted_gpc)
print('knn roc -',roc_knn)
print('rfc roc -',roc_rfc)
print('gpc roc -',roc_gpc)

f1_knn=f1_score(valid_df_cancer['Cancer'],predicted_knn)
f1_rfc=f1_score(valid_df_cancer['Cancer'],predicted_rfc)
f1_gpc=f1_score(valid_df_cancer['Cancer'],predicted_gpc)
print('knn f1 -',f1_knn)
print('rfc f1 -',f1_rfc)
print('gpc f1 -',f1_gpc)



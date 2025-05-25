import os
import sys
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath("util"))

import util.img_util as loadI
from util.feature_A import feature_A
from util.feature_B import feature_B
from util.feature_C import feature_C
from feature_H import feature_H
from util.classifier import classifier

def path_finder():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    relative_path_to_data = os.path.join(current_directory, './data')
    data_folder_path = os.path.normpath(relative_path_to_data)
    return data_folder_path

test = loadI.ImageDataLoader(path_finder())

testBatch = test.__iter__()

df = pd.DataFrame()

imgIDx = 0

csv_dir = os.path.dirname(os.path.abspath(__file__)) + '/' + 'metadata.csv'

df_ground=pd.read_csv(csv_dir)

df_ground['cancer']=np.where( (df_ground[ 'diagnostic']=='BCC') ^ (df_ground['diagnostic']=='MEL') ^ (df_ground['diagnostic']=='SCC'),1,0)

df_annotations= pd.read_csv(r"C:\Users\anna-\OneDrive\Desktop\ITU\2nd semester\Projects in Data Science\Project\Manual Annotations - Sheet1.csv")


for imgAndMask in testBatch:
    img = imgAndMask[0]
    mask = imgAndMask[1]
    patNumber = imgAndMask[2]

    hairy=df_annotations['Average'][(df_annotations['img_id']==patNumber)].to_numpy()
    cancer_part=df_ground['cancer'][( df_ground['img_id']==patNumber )].to_numpy()

    feature_A(mask,imgIDx,df)
    feature_B(img, mask, imgIDx, df)
    feature_C(img, imgIDx, df)
    feature_H(img, imgIDx, df)

    df.loc[imgIDx,'Cancer']=cancer_part[0]

    imgIDx += 1

df_new=pd.DataFrame()
df_new['manual'] = df_annotations['Average']
df_new['model'] = df['Hair']

def zero_two(x):
    if x<0.01:
        return 0
    elif x<0.1:
        return 1
    else:
        return 2
    
df_new['transformed']= df_new['model'].apply(zero_two)

## Training and testing data NOTE: stratify with equal amount of men and women
from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(5,shuffle=True)

## Split into features and cancer values
x=df.drop(columns='Cancer')
y=df['Cancer']

## Split based on cancer values FIXME Testing set seed
for train_idx, test_idx in skf.split(x, y):
    train_df = df.iloc[train_idx]  # 80%
    test_df = df.iloc[test_idx]    # 20%
    break
## Gives training data (80%) and testing data (20%)

## Training and validation
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4)

## Split into features and cancer values
x = train_df.drop(columns='Cancer')
y = train_df['Cancer']

## Split based on cancer values
for train_idx, test_idx in sss.split(x, y):
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

classifier(train_df_scaled, train_df_cancer, valid_df_cancer, valid_df_scaled)


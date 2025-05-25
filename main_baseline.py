import os
import sys
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath("util"))

#Importing the loader class, features and classifier.
import util.img_util as loadI
from util.feature_A import feature_A
from util.feature_B import feature_B
from util.feature_C import feature_C
from util.classifier import classifier

#This function is used to find the path where the data is kept, i.e. the picture and masks for the picture(if there are any)
#It's done this way so we avoid hard coded paths and so it runs without any user modifications
def path_finder():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    relative_path_to_data = os.path.join(current_directory, './data')
    data_folder_path = os.path.normpath(relative_path_to_data)
    return data_folder_path

#List of images, masks and image names
loaderclass = loadI.ImageDataLoader(path_finder())

#Initialize DataFrame for features
imageBatch = loaderclass.__iter__()

#Initialize DataFrame for features
df = pd.DataFrame()

imgIDx = 0

#Code for finding the path of the CSV
#It's done this way so we avoid hard coded paths and so it runs without any user modifications
metadata_dir = os.path.dirname(os.path.abspath(__file__)) + '/' + 'metadata.csv'

#Loading the CSV
df_ground=pd.read_csv(metadata_dir)

df_ground['cancer']=np.where( (df_ground[ 'diagnostic']=='BCC') ^ (df_ground['diagnostic']=='MEL') ^ (df_ground['diagnostic']=='SCC'),1,0)

#We use this for to apply the features to the picture and to get a dataframe which includes all of the data returned from them
for imgAndMask in imageBatch:
    img = imgAndMask[0]
    mask = imgAndMask[1]
    patNumber = imgAndMask[2]

    cancer_part=df_ground['cancer'][( df_ground['img_id']==patNumber )].to_numpy()

    feature_A(mask,imgIDx,df)
    feature_B(img, mask, imgIDx, df)
    feature_C(img, imgIDx, df)

    df.loc[imgIDx,'Cancer']=cancer_part[0]

    imgIDx += 1

## Training and testing data NOTE: random state to keep the test always the same
from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(5,shuffle=True,random_state=42)

## Split into features and cancer values
x=df.drop(columns='Cancer')
y=df['Cancer']

## Split based on cancer values
for train_idx, test_idx in skf.split(x, y):
    train_df = df.iloc[train_idx]  # 80%
    test_df = df.iloc[test_idx]    # 20%
    break
## Gives training data (80%) and testing data (20%)

## Training and validation
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

## Split into features and cancer values
x = train_df.drop(columns='Cancer')
y = train_df['Cancer']

## Split based on cancer values
for train_idx, test_idx in sss.split(x, y):
    train_df = df.iloc[train_idx]  # 80%
    valid_df = df.iloc[test_idx]    # 20%
## Takes previous training data
## Gives training data (80%) and validation data (20%)

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


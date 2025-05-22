import os
from mask_function import get_mask
import cv2
import pandas as pd
import numpy as np
from feature_A import feature_A
from feature_B import f_feature_B
from feature_C import feature_C

## Fetch files in directories, create base paths
images=os.listdir(r'C:\Users\filip\Dokumenty\Data Science 2nd semester\Project\my_project_prep\baseline model - ABC features\imgs_part_1\imgs_part_1')
masks=os.listdir(r'C:\Users\filip\Dokumenty\Data Science 2nd semester\Project\my_project_prep\baseline model - ABC features\lesion_masks')
image_path=r'C:\Users\filip\Dokumenty\Data Science 2nd semester\Project\my_project_prep\baseline model - ABC features\imgs_part_1\imgs_part_1'
mask_path=r'C:\Users\filip\Dokumenty\Data Science 2nd semester\Project\my_project_prep\baseline model - ABC features\lesion_masks'

## Remove duplicate masks
images_set=set(images)
masks=[mask for mask in masks if mask[:-9]+'.png' in images_set]

## Load CSV and create a column for presence of cancer
df_ground=pd.read_csv(r"C:\Users\filip\Dokumenty\Data Science 2nd semester\Project\my_project_prep\baseline model - ABC features\metadata.csv")
df_ground['cancer']=np.where( (df_ground[ 'diagnostic']=='BCC') ^ (df_ground['diagnostic']=='MEL') ^ (df_ground['diagnostic']=='SCC'),1,0)

## Create empty df for features
df=pd.DataFrame()


## Main loop through pictures
image_idx = 0
mask_idx = 0
has_mask = True

while image_idx < 200: # and mask_idx < len(masks): ## len(images)
    image = images[image_idx]
    mask = masks[mask_idx]
    
    expected_im = mask[:-9] + '.png' ## Remove _mask from mask name, compare to image

    if image == expected_im: ## If matching, load both
        has_mask = True
        mask_idx += 1  ## Move to next mask only if match
        image=cv2.imread(f'{image_path}/{image}',cv2.IMREAD_COLOR_RGB)
        mask=cv2.imread(f'{mask_path}/{mask}',cv2.IMREAD_GRAYSCALE)
        if np.sum(np.nonzero(mask))==0: ## If mask black image
            mask=get_mask(image) ## Make own mask
            mask = mask.astype(int)

    else: ## If not matching, load image and create mask
        has_mask = False
        image=cv2.imread(f'{image_path}/{image}',cv2.IMREAD_COLOR_RGB)
        mask=get_mask(image)
        mask = mask.astype(int)


    ## Mask image with mask
    image_masked=image.copy()
    image_masked[mask==0] = 0

    ## Extract features from images/masks
    feature_A(mask,image_idx,df)
    f_feature_B(image,mask,image_idx,df)
    feature_C(image_masked,image_idx,df)
    image_idx += 1  ## Always move to the next image

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


## Train a classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knntrained = knn.fit(train_df_scaled, train_df_cancer['Cancer']) ## Training data and cancer values

## Primitive way of finding accuracy
from sklearn.metrics import accuracy_score
predited=knntrained.predict(valid_df_scaled)
acc_knn = accuracy_score(valid_df_cancer['Cancer'], predited)
print(f'{acc_knn*100}%')

## ~50%, not great
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import numpy as np

df_manual=pd.read_csv('Manual Annotations - Sheet1.csv')        # read CSV with manual annotation
df_model=pd.read_csv('model_features.csv')                      # read CSV with model's annotations
df_img_id=pd.read_csv('features2.csv')                          # read image IDs 

df_model.insert(0, 'id', df_img_id['ID'].values)                # merge image IDs onto the model's results (they are in the same order)

df_model_hair = df_model.drop(columns=['A - worst asymmetry', 'B - compactness', 'C - red variance', 'C - green variance', 'C - blue variance', 'Cancer'])  #drop columns we are not interested in


def zero_two(x):            # change hair rating from percentage to 0, 1, 2 like in the manual annotations
    if x<0.01:
        return 0
    elif x<0.1:
        return 1
    else:
        return 2  
df_model_hair['Hair']= df_model_hair['Hair'].apply(zero_two)

merged = df_model_hair.merge(               #merge the manual and model annotations onto one data frame
    df_manual[['img_id', 'Average']],
    left_on='id',
    right_on='img_id',
    how='left'
).drop(columns='img_id')

merged = merged.rename(columns={
    'Hair': 'Model Score',
    'Average': 'Manual Score'
})

cohen_kappa_score(merged['Manual Score'], merged['Model Score'])        #calculate Cohen's Kappa score to see how much our manual annotations agree with the model's annotations
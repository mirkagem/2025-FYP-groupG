import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

#Combined model paths, done like this to avoid hardcoded paths
df_images = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+"/util/model_features.csv")
df_images2 = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+"/util/img_id.csv")
df_metadata = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+"/dataset.csv")

print(df_metadata.shape) #ImageID of removed image: PAT_987_1859_859
print(df_images.shape)
df_images['img_id'] = df_images2['ID']
combined_df = pd.merge(df_images, df_metadata, on='img_id', how='inner')
print(combined_df.shape)
combined_df = combined_df.dropna()
# combined_df2 = combined_df.copy() - to be removed

#Data cleaning to be suitable for model
combined_df.replace("True", 1, inplace=True)
combined_df.replace("False", 0, inplace=True)
combined_df.replace(True, 1, inplace=True)
combined_df.replace(False, 0, inplace=True)
combined_df.replace("UNK", float("NaN"), inplace=True)

combined_df = combined_df.drop(columns=['biopsed','img_id',
      'has_piped_water',
      'has_sewage_system', 'fitspatrick', 'region', 'diagnostic'])

print(combined_df.columns)
combined_df.head(5)

#This cell is used to find the importance of each feature
features_to_include = ['A - worst asymmetry', 'B - compactness', 'C - red variance',
       'C - green variance', 'C - blue variance', 
       'bleed', 'elevation', 'grew', 'itch', 'diameter_2']

y = combined_df['Cancer_x'].copy()
x = combined_df[features_to_include]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=101) #stratify = y makes sure that both training and test sets preserve the original class proportions.

print(f'x_train : {x_train.shape}')
print(f'y_train : {y_train.shape}')
print(f'x_test : {x_test.shape}')
print(f'y_test : {y_test.shape}')

rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

#This cell is used to find the best parameters for RFC 
param_grid = [{
    'n_estimators':[100,200,500], #More trees = better performance up to a point, then diminishing returns.
    'max_depth': [3, 5, 10, 20], #higher the values the more we risk overfitting
    'criterion': ['entropy', 'gini'], #can also comment this out
    'min_samples_split': [5,10,15], # minimum number of samples a node must have to be split into two child nodes in a decision tree. Default is 2 (canm overfit)
    'min_samples_leaf': [1, 2, 4]#,
    #'max_features': ['sqrt', None]

}]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)
grid_search = GridSearchCV(rf_model,
                           param_grid,
                           cv=skf, #this splits testing data to test and validation based on cancer/non-cancer proportions (so they have same percentage of cancer)
                                    #if we use this, we dont have to split training data ourselves at the beginning
                           scoring = 'recall',
                           n_jobs=-1,                          
                           )

grid_search.fit(x_train, y_train)

print(f'Grid search best score: {grid_search.best_score_}')
print(f'Grid search best parameters: {grid_search.best_params_}')

#Training the rf with what criteria recomended by grid search (= best parameters)
best_params = grid_search.best_params_

# Grid search best score: 0.9655566502463054
# Grid search best parameters: {'criterion': 'gini', 'max_depth': 3, 'min_samples_leaf': 2, 'min_samples_split': 15, 'n_estimators': 100}

#Training the rf with what criteria recomended by grid search (= best parameters)
best_params = grid_search.best_params_

#This cell shows the detailed results while using the best parameters
rf_final = RandomForestClassifier(**best_params, random_state=101) 
rf_final.fit(x_train, y_train)

#Evaluating on training and test set
y_pred = rf_final.predict(x_train)
print("Train Report:")
print(classification_report(y_train, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred))

y_test_pred = rf_final.predict(x_test)
print("\nTest Report:")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

print(best_params)

features_to_include = ['A - worst asymmetry', 'B - compactness', 'C - red variance',
       'C - green variance', 'C - blue variance','grew']
                                                               #, 'bleed', 'elevation', 'diameter_2','itch']
print(features_to_include)

y = combined_df['Cancer_x'].copy()
x = combined_df[features_to_include]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=101)

rf_final = RandomForestClassifier(**best_params, random_state=101) 
rf_final.fit(x_train, y_train)


print(f'Performance for model trained on the following features {features_to_include}:')

y_pred = rf_final.predict(x_train)
print("Train Report:")
print(classification_report(y_train, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred))

y_test_pred = rf_final.predict(x_test)
print("\nTest Report:")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

importances=rf_model.feature_importances_
std=np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)

print(best_params)


features_to_include = ['A - worst asymmetry', 'B - compactness', 'C - red variance',
       'C - green variance', 'C - blue variance','grew', 'bleed']
                                                               #, 'bleed', 'elevation', 'hurt', 'diameter_2','itch']
print(features_to_include)

y = combined_df['Cancer_x'].copy()
x = combined_df[features_to_include]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=101)

rf_final = RandomForestClassifier(**best_params, random_state=101) 
rf_final.fit(x_train, y_train)


print(f'Performance for model trained on the following features {features_to_include}:')

y_pred = rf_final.predict(x_train)
print("Train Report:")
print(classification_report(y_train, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred))

y_test_pred = rf_final.predict(x_test)
print("\nTest Report:")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

importances=rf_model.feature_importances_
std=np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)

print(best_params)

features_to_include = ['A - worst asymmetry', 'B - compactness', 'C - red variance',
       'C - green variance', 'C - blue variance','grew', 'bleed', 'elevation']
                                                               #'diameter_2','itch']
print(features_to_include)

y = combined_df['Cancer_x'].copy()
x = combined_df[features_to_include]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=101)

rf_final = RandomForestClassifier(**best_params, random_state=101) 
rf_final.fit(x_train, y_train)


print(f'Performance for model trained on the following features {features_to_include}:')

y_pred = rf_final.predict(x_train)
print("Train Report:")
print(classification_report(y_train, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred))

y_test_pred = rf_final.predict(x_test)
print("\nTest Report:")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

importances=rf_model.feature_importances_
std=np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)

print(best_params)

features_to_include = ['A - worst asymmetry', 'B - compactness', 'C - red variance',
       'C - green variance', 'C - blue variance','grew', 'bleed', 'elevation','itch']
print(features_to_include)

y = combined_df['Cancer_x'].copy()
x = combined_df[features_to_include]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=101)

rf_final = RandomForestClassifier(**best_params, random_state=101) 
rf_final.fit(x_train, y_train)


print(f'Performance for model trained on the following features {features_to_include}:')

y_pred = rf_final.predict(x_train)
print("Train Report:")
print(classification_report(y_train, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred))

y_test_pred = rf_final.predict(x_test)
print("\nTest Report:")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

importances=rf_model.feature_importances_
std=np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)

print(best_params)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

#Reads the data, this is done this way to avoid hardcoding the paths
df = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+"/dataset.csv")

y = df['diagnostic'].copy() #column with verdict: cancer or not cancer
best_features = ['itch','grew','bleed','elevation', 'diameter_2']
#best_features = 
x = df[best_features]

#Deviding into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=101) #stratify = y makes sure that both training and test sets preserve the original class proportions.

#This gives us:
#80% training (x_train)
#20% test (x_test)

print(f'x_train : {x_train.shape}')
print(f'y_train : {y_train.shape}')
print(f'x_test : {x_test.shape}')
print(f'y_test : {y_test.shape}')

rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

param_grid = [{
    'n_estimators':[100,200,500], #More trees = better performance up to a point, then diminishing returns.
    'max_depth': [3, 5, 10, 20], #higher the values the more we risk overfitting
    'criterion': ['entropy', 'gini'], #can also comment this out
    'min_samples_split': [5,10,15], # minimum number of samples a node must have to be split into two child nodes in a decision tree. Default is 2 (canm overfit)
    'min_samples_leaf': [1, 2, 4]#,
    #'max_features': ['sqrt', 'log2', None]

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

rf_final = RandomForestClassifier(**best_params, random_state=101) #if we want to increase recall for 0 (lower false positive), add here: class_weight='balanced'
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

#If we wanted to change the recall for healthy people, we could adjust the threshold for classifying something as cancer - but it didnt make much difference in this model (1 percent better recall on 0)
probs_val = rf_final.predict_proba(x_train)[:, 1]

thresholds = np.arange(0.3, 0.91, 0.05)

print("Threshold\tRecall_0\tRecall_1\tTotal_Accuracy")
print("-" * 50)

best_thresh = 0.5
best_recall_1 = 0
best_report = ""

for thresh in thresholds:
    preds = (probs_val >= thresh).astype(int)
    recall_0 = recall_score(y_train, preds, pos_label=0)
    recall_1 = recall_score(y_train, preds, pos_label=1)
    accuracy = (preds == y_train).mean()
    
    print(f"{thresh:.2f}\t\t{recall_0:.2f}\t\t{recall_1:.2f}\t\t{accuracy:.2f}")
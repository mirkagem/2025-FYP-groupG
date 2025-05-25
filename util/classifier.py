def classifier(train_df_scaled, train_df_cancer, valid_df_cancer, valid_df_scaled,test_df_scaled, test_df_cancer):
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

    ## Different ways of evaluating the models
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,roc_auc_score,f1_score, cohen_kappa_score
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

    cohen_kappa_knn = cohen_kappa_score(valid_df_cancer['Cancer'], predicted_knn)
    cohen_kappa_rfc= cohen_kappa_score(valid_df_cancer['Cancer'], predicted_rfc)
    cohen_kappa_gpc= cohen_kappa_score(valid_df_cancer['Cancer'], predicted_gpc)
    print('knn cohen kappa -', cohen_kappa_knn)
    print('rfc cohen kappa -', cohen_kappa_rfc)
    print('gpc cohen kappa -', cohen_kappa_gpc)
    print()
    ## Testing

    predicted_knn_test=knntrained.predict(test_df_scaled)
    predicted_rfc_test=rfctrained.predict(test_df_scaled)
    predicted_gpc_test=gpctrained.predict(test_df_scaled)

    acc_knn = accuracy_score(test_df_cancer['Cancer'], predicted_knn_test)
    acc_rfc = accuracy_score(test_df_cancer['Cancer'], predicted_rfc_test)
    acc_gpc = accuracy_score(test_df_cancer['Cancer'], predicted_gpc_test)
    print('knn -',acc_knn)
    print('rfc -',acc_rfc)
    print('gpc -',acc_gpc)

    recall_knn = recall_score(test_df_cancer['Cancer'],predicted_knn_test)
    recall_rfc = recall_score(test_df_cancer['Cancer'],predicted_rfc_test)
    recall_gpc = recall_score(test_df_cancer['Cancer'],predicted_gpc_test)
    print('knn recall -',recall_knn)
    print('rfc recall -',recall_rfc)
    print('gpc recall -',recall_gpc)

    roc_knn=roc_auc_score(test_df_cancer['Cancer'],predicted_knn_test)
    roc_rfc=roc_auc_score(test_df_cancer['Cancer'],predicted_rfc_test)
    roc_gpc=roc_auc_score(test_df_cancer['Cancer'],predicted_gpc_test)
    print('knn roc -',roc_knn)
    print('rfc roc -',roc_rfc)
    print('gpc roc -',roc_gpc)

    f1_knn=f1_score(test_df_cancer['Cancer'],predicted_knn_test)
    f1_rfc=f1_score(test_df_cancer['Cancer'],predicted_rfc_test)
    f1_gpc=f1_score(test_df_cancer['Cancer'],predicted_gpc_test)
    print('knn f1 -',f1_knn)
    print('rfc f1 -',f1_rfc)
    print('gpc f1 -',f1_gpc)
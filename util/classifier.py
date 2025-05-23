def classifier(train_df_scaled, train_df_cancer, valid_df_cancer, valid_df_scaled):
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

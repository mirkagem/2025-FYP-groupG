def classifier(train_df_scaled, train_df_cancer):
    #The fine tuned parameters for RFC
    parameters_rfc ={'class_weight': 'balanced', 'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100,'max_features': 'sqrt'}
    ## Train the KNN classifier
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 9)
    knntrained = knn.fit(train_df_scaled, train_df_cancer['Cancer']) ## Training data and cancer values

    ##T Train the RFC classifier
    from sklearn.ensemble import RandomForestClassifier
    rfc=RandomForestClassifier(**parameters_rfc)
    rfctrained=rfc.fit(train_df_scaled,train_df_cancer['Cancer'])

    return knntrained, rfctrained
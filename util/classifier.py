def classifier(train_df_scaled, train_df_cancer):
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

    return knntrained, rfctrained, gpctrained
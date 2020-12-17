# -*- coding: utf-8 -*-
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import VarianceThreshold

'''
Function Goal: Center and scale features without modifying label
Function Input: df: contains label and features
Function Output: df_normalized: df of normalized features
'''
def scenter_scale_features(df):
    
    ## list of features
    features = df.columns[1:]
    
    ## Normalize features
    df[features] = (df[features]-df[features].mean())/df[features].std()
    
    return df


'''
Function Goal: Power transform skewed featuers using box-cox method
Function Input: df: contains label and features
                skew_threshold: sets the skew_threshold which determines which features are to be transformed
                x_shift: sets the shift in order to do a log(x + x_shift) transformation to avoid log(0)
Function Output: df: contains label with power transformed features
                 pt: fitted power transform object, to be used on transforming new data
'''
def power_transform_skewed_features(df, skew_threshold=2, x_shift=1):
    
    ## Calculate skew and filter by skew threshold
    skewed = df.skew()
    skewed_features = skewed[(skewed > skew_threshold ) | (skewed < -skew_threshold)]
    
    ## Create a copy dataframe to make transformations on
    df_transform = df.copy()
    
    ## Shift data by 1 to avoid log(0)
    df_transform[skewed_features.index] = df_transform[skewed_features.index] + x_shift
    
    ## Fit power transformer
    pt = PowerTransformer(method='box-cox', standardize=False).fit(df_transform[skewed_features.index])
    
    ## Power transform skewed features
    df[skewed_features.index] = pt.transform(df_transform[skewed_features.index])
    
    return df, pt

'''
removes features of low variance
'''
def low_var_exclusion(df, treshold):
    
    label = df["Label"]
    selector = VarianceThreshold(treshold)
    features=selector.fit_transform(df.drop('Label', axis=1))
    index=selector.get_support(indices=True)

    index_df=list(range(0, df.shape[1], 1))

    new=df
    
    for i in index_df:
        if (not (i in index)):
            if (not (df.columns[i] == 'Label')): new=new.drop(df.columns[i], axis=1)
    return new
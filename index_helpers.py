# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd


'''
Example: How to use
df = read_and_merge_segmented_data()
df = index_df_by_person(df)
X_train, X_test, y_train, y_test = train_test_split_on_index(features = df.drop("Label", axis=1), 
                                                             label = df["Label"])

'''

'''
Function Goal: Reads segmented labels and features and merge into dataframe - optionally include data
Function Input: fine_segmentation: True = import fine segmentation data - False = import coarse segmentation data
                exlude_expert: True = excludes column for which expert labeled the data
                exlude_meta_data: True = exludes meta data
Function Output: df - merged DataFrame
                
'''
def read_and_merge_data(segmentation=True, fine_segmentation=True, exlude_expert=True, exclude_meta_data=True):
    
    if segmentation:
      ## Set path
      if(fine_segmentation):
          PATH_labels = Path('../data/labels_fine_segmentation.csv')
          PATH_features = Path('../data/features_fine_segmentation.csv')
      else:
          PATH_labels = Path('../data/labels_coarse_segmentation.csv')
          PATH_features = Path('../data/features_coarse_segmentation.csv')
    else:
      PATH_features = Path("../data/features_no_segmentation.csv")
      PATH_labels = Path("../data/labels_no_segmentation.csv")

    ## Read data
    labels_segmentation = pd.read_csv(PATH_labels, header=0, index_col=0)
    features_raw = pd.read_csv(PATH_features, header=0, index_col=0)
    
    ## save column names
    X_col_all = features_raw.columns.to_list()
    
    ## Translate floats of categorical to int
    labels_segmentation["Label"] = labels_segmentation["Label"].astype(int)
    
    ## Drop "File_Name" from features_raw because labels already has it. Will be merged
    features_segmentation = features_raw.drop('File_Name', axis=1)
    
    ## Merge features and labels to same dataset
    df = pd.merge(labels_segmentation, features_segmentation, left_index=True, right_index=True)
    
    ## List of features to exlude from the features dataframe
    drop_list = []
    if(exlude_expert):
        drop_list = drop_list + ['Expert']
    if(exclude_meta_data):
        drop_list = drop_list + ['Gender', 'Resp_Condition', 'Symptoms']
    
    ## List of categorical features
    categorical_feat = ['Expert', 'Gender', 'Resp_Condition', 'Symptoms']
    
    ## Drop labels from list
    if(len(drop_list) != 0):
        df = df.drop(drop_list, axis=1)
        categorical_feat = [e for e in categorical_feat if e not in drop_list]
        
    numerical_feat = [e for e in df.columns.to_list() if e not in categorical_feat]  
        
    return df, categorical_feat, numerical_feat[2:], X_col_all[1:]


'''
Function Goal: Create hierarchical index based on column "File_Name", splitted by person and n_recording
Function Input: df: DataFrame to index
Function Output: df: Indexed DataFrame
P.S: Keep labels and features in same dataframe to avoid having to match later. numerical indices 
are removed with this function
'''
def index_df_by_person(df):
    
    ## Create new dataframe split by '_', one column per value
    File_Name_split = df["File_Name"].str.split('_', expand=True)
    
    ## Rename columns - Done to improve clarity of index headers in the final df
    File_Name_split.rename(columns={0:'File_Name_split', 1:'File_n_recording'}, inplace=True)
    
    ## Merge in splitted columns based on their numerical index
    df = pd.merge(df, File_Name_split, left_index=True, right_index=True)
    
    count = df['File_Name_split'].value_counts()
    
    ## Create hierarchical index based on splitted column
    df = df.set_index(['File_Name_split', 'File_n_recording'])
    
    ## Drop splitted column
    df = df.drop('File_Name', axis=1)
    
    return df, count


'''
Function Goal: Create a training and testing set based on index - USed here to split sets by people
Function Input: features: Dataframe of features
                label: DataFrame of labels 
                level: Which level of the hierarchical index to split on. Default is 0
                test_size: size of test set
                random_state: int, for reproducability
Function Output: X_train, X_test, y_train, y_test - split by index

'''
def train_test_split_on_index(features, label, level=0, test_size=0.2, random_state = 42):
    
    ## Split indices into training and testing
    X_train_ind, X_test_ind, y_train_ind, y_test_ind = train_test_split(features.index.levels[level], 
                                                                        label.index.levels[level],
                                                                        test_size=test_size, 
                                                                        random_state=random_state)
    
    ## Slice features by split indices (persons)
    X_train = features.loc[X_train_ind]
    X_test = features.loc[X_test_ind]
    y_train = label.loc[y_train_ind]
    y_test = label.loc[y_test_ind]
    return X_train, X_test, y_train, y_test

'''
Function Goal: translate categorical features from float to int
'''
def categorical_float_to_int(df):
    categorical_features = df.drop('Label',axis=1).columns[:19]
    df[categorical_features] = df[categorical_features].astype(int)
    return df

'''
Function Goal: Take columns starting with EEPD and translate them to dummy variables
'''
def categorical_to_dummy(df, cat_list):
    df = pd.get_dummies(df, columns=cat_list)
    return df

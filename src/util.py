# -*- coding: utf-8 -*-
"""
Utility functions for use in poverty predection project

Created on Wed Jul 13 17:54:38 2022

@author: youse
"""

import numpy as np
import pandas as pd 
from config import (OUTLIER_METHOD, MANUAL_OUTLIER_COLS, IMPUTE_METHOD)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer ,IterativeImputer


def decode_columns(df_data, df_codebook):
   '''
   The dataset from the kaggle challenge contains encoded column names. This
   function returns the df with the decoded column names.

    Parameters
    ----------
    df_data : DataFrame
        training or test set
    df_codebook : DataFrame
        kaggle-provided column mappings, from 'codebook.csv'

    Returns
    -------
    df_decoded

    '''
    
   # TODO when doing this for test set, make sure columns are in the same order as train
    
   # decode column names and replace in df
   columns_encoded = pd.Series(df_data.columns)
   column_map = dict(zip(df_codebook['Variable name'], df_codebook['Variable description']))
   columns_decoded = columns_encoded.map(column_map)
    
   # some columns didn't match, do these contain useful data?
   df_column_map = pd.DataFrame(zip(columns_encoded, columns_decoded))
   df_column_map.columns = ['Encoded', 'Decoded']
   nan_cols = df_column_map[df_column_map.Decoded.isna()]
   df_nan_col_names = df_data[nan_cols.Encoded.values]
    
   # for non-matched columns, replace with original/encoded column name
   df_column_map['columns_final'] = np.where(df_column_map.Decoded.isna(), df_column_map.Encoded, df_column_map.Decoded)
   df_data.columns = df_column_map['columns_final'].values
   
   return df_data


#########################
# PREPROCESSING FUNCTIONS
#########################


def drop_redundant_cols(df, ratio_corr=0.95):
    '''
    drop (almost) perfectly correlated columns, defined by ratio_corr

    Parameters
    ----------
    df : DataFrame
    ratio_corr : int
        how correlated must cols be to be considered redundant

    Returns
    -------
    df
    '''

    # remove redundant (perfectly correlated) features
    df_corr = df.corr()
    df_upper_triangle = df_corr.where(np.triu(np.ones(df_corr.shape),k=1).astype(np.bool))
    redundant_cols = [column for column in df_upper_triangle.columns if any(df_upper_triangle[column] > ratio_corr)]
    
    # manual review of redundant columns
    # TODO: remove need for manual review here
    # df_redundant_corr = df_upper_triangle[redundant_cols]
    # df_redundant_train = df[redundant_cols]
    
    # drop redundant columns
    df.drop(columns=['# of total individuals in the household', 'Age squared', 'size of the household', 'hogar_total squared'], inplace=True)
    
    return df


def drop_mostly_missing(df, col_ratio=90, row_ratio=60): 
    '''
    Drop columns and rows that have "too many" missing values, defined by 
    col_ratio and row_ratio

    Parameters
    ----------
    df : DataFrame
    col_ratio : int, optional
        Tolerance limit (%) for missing column values. The default is 90.
    row_ratio : int, optional
        Tolerance limit (%) for missing row values. The default is 60.

    Returns
    -------
    df
    '''
    
    print("-- Dropping columns and rows with large amounts of missing data --")
    
    # per column/feature
    df_ratio_na_col = df.isnull().sum()/len(df) * 100
    df_ratio_na_col.sort_values(inplace=True, ascending=False)
    df_mostly_na_col = df[df_ratio_na_col[df_ratio_na_col >= 90].index]
    df = df.drop(columns = df_mostly_na_col.columns) # we don't end up dropping any columns at the 90% missing threshold
    print(f'Dropped cols: {df_mostly_na_col.columns.to_list()}')
    
    # per row/observation
    df_ratio_na_row = df.notna().sum(axis=1) / df.shape[1] * 100
    len_pre_filter = len(df)
    ix_to_keep = df_ratio_na_row[df_ratio_na_row >= row_ratio].index
    len_post_filter = len(ix_to_keep)
    df = df.filter(items=ix_to_keep, axis=0)
    assert(len(df) == len_post_filter)
    print(f"Dropped {len_pre_filter - len_post_filter} rows")
    
    return df


def check_regular_bool(unique_col_values):
    '''
    Helper function to make sure all elements in boolean column don't contain anything except 1, 0, or nan
    '''
    
    return set(unique_col_values) - {1,0,np.nan} == set()


def clean_boolean(df):
    '''
    Ensure presence of only 1, 0, or nan in boolean columns, then convert to type boolean

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    df : DataFrame
    '''
    
    print("-- Cleaning boolean columns --")

    # ensure boolean columns have only up to 3 values: 1, 0, and nan
    bool_cols = [col for col in df.columns if '=1' in col]
    bool_cols.append('=2 zona rural')
    bool_cols.append('owns a tablet')
    df[bool_cols] = df[bool_cols].astype(int)
    df_bools = df[bool_cols].nunique()
    df_bools.sort_values(inplace=True, ascending=False)
     
    # flag any columns that have more than 3 values
    df_irregular_bool = df[bool_cols].apply(lambda x: check_regular_bool(x.unique()))
    irregular_bool_cols = df_irregular_bool[df_irregular_bool != True].index
    
    # drop this column because all the observations are 0 (no variance)
    df.drop(columns='=1 if rubbish disposal mainly by throwing in river, creek or sea', inplace=True)
    bool_cols.remove('=1 if rubbish disposal mainly by throwing in river, creek or sea')
   
    if len(irregular_bool_cols > 0):
        print(f"The following boolean columns contain irregular values: {irregular_bool_cols}")
    else:
        print("All boolean columns contain only 1, 0, or nan")
    
    # Convert boolean to True/False
    df[bool_cols] = df[bool_cols].astype(bool)
    
    return df


def clean_mixed_cols(df, method='bool'):
    '''
    Some columns contains a mix of 'yes', 'no', and int values. In order
    to retain information, three methods are used to address this inconsistency
    with the intention of comparing model results using each respective method

    Parameters
    ----------
    df : DataFrame
    method : str, optional
        Which method to use for handling mixed values:
             'bool' - treat no as False, and yes and int values as True
             'exclude_yes' - remove 'yes' values and set 'no' to 0
             'exclude_string' - drop all string values
        The default is 'bool'.

    Returns
    -------
    df : DataFrame
    '''
    
    print('-- Cleaning columns with mixed dtypes --')
    
    # TODO check which of these methods leads to the best performance
    mixed_cols = ['years of education of female head of household', 'years of education of male head of household', 'Dependency rate']
    
    if method == 'bool':
        for col in mixed_cols:
            df[col] = df[col].apply(lambda x: False if 'no' in str(x) else True)
        df[mixed_cols] = df[mixed_cols].astype(bool)
        df.drop(columns='dependency squared', inplace=True) # if bool, squaring this col makes no sense
        
    elif method == 'exclude_yes':
        for col in mixed_cols:
            df[col] = df[col].apply(lambda x: 0 if 'no' in str(x) else (np.nan if 'yes' in str(x) else x))
        df[mixed_cols] = df[mixed_cols].astype(float)
   
    elif method == 'exclude_string':
        for col in mixed_cols:
            df[col] = df[col].apply(lambda x: np.nan if str(x).strip() in ['yes', 'no'] else x)
        df[mixed_cols] = df[mixed_cols].astype(float)
    
    return df


def handle_outliers(df, manual_outlier_cols=None, method='drop', bound_coeff=3):
    '''
    Detects and handles outliers as specifed by argument 'method'

    Parameters
    ----------
    df : DataFrame
    manual_outlier_cols : list, optional
        List of columns to check for outliers in. The default is None, which will
        check all columns for outliers.
    method : str, optional
        How to handle outliers, options are 'drop' or 'cap'. The default is 'drop'.
    bound_coeff : int, optional
        Used to calculate bounds of acceptable values, outside of which a value is
        considered an outlier. A larger bound_coeff means more tolerance. The default is 3.

    Returns
    -------
    df : DataFrame
    '''
    
    print('-- Handling outliers --')
    numeric_columns = df.select_dtypes(include=["float64", 'int64']).columns

    # check all columns or only specified columns for outliers
    if manual_outlier_cols is not None:
        df_numeric = df[manual_outlier_cols]   
    else:    
        df_numeric = df[numeric_columns]
    
    # calculate IQR
    q1 = df_numeric.quantile(0.25)
    q3 = df_numeric.quantile(0.75)
    IQR = q3 - q1
    
    # calculate bounds outside of which a data point is considered an outlier
    upper_bound = q3 + bound_coeff * IQR
    lower_bound = q1 - bound_coeff * IQR
    
    # flag outliers
    df_outliers = df_numeric[(df_numeric < lower_bound) | (df_numeric > upper_bound)]
    outlier_cols = (df_outliers.notna().any()[df_outliers.notna().any()]).index # get only columns with an outlier present
    #df_outliers = df_outliers[outlier_cols]
    ix_outliers = (df_outliers[df_outliers.notna().any(axis=1)]).index
    
    print(f"Identified {len(outlier_cols)} column(s) with outliers")
    print(f"Found {len(ix_outliers)} outliers")
    
    # handle outliers
    
    # this method leads to a loss of information
    if method=='drop':
        print("Dropping outliers")
        df.drop(index=ix_outliers, inplace=True)
    
    # this method assumes that past a certain rent amount, no new behavior is observed
    elif method=='cap':
        print("Capping outliers to bounding values")
        for col in outlier_cols:
            df[col] = np.where(df[col] > upper_bound[col],
                          upper_bound[col],
                          np.where(df[col] < lower_bound[col],
                              lower_bound[col],
                              df[col]))
    
    # TODO: code this functionality if the first two approaches aren't working well
    # elif outlier_method=='impute':
        
    return df


def clean_numeric(df, outlier_method=OUTLIER_METHOD, manual_outlier_cols=MANUAL_OUTLIER_COLS):
    '''
    Drop redundant numeric columns, set numeric data types, and handle outliers

    Parameters
    ----------
    df : DataFrame
    manual_outlier_cols : list, optional
        List of columns to check for outliers in. The default is None, which will
        check all columns for outliers.
    method : str, optional
        How to handle outliers, options are 'drop' or 'cap'. The default is 'drop'.
    

    Returns
    -------
    df : DataFrame
    '''
    
    print('-- Cleaning numeric columns --')
         
    # manually check object columns for should-be numeric cols
    # df_obj = df.select_dtypes("object")
    
    # squared columns seem redundant, I'll drop for now, may add seperate handling method later
    squared_cols = [col for col in df.columns if 'squared' in col]
    df.drop(columns=squared_cols, inplace=True)
    
    # manually inspect numeric columns
    numeric_columns = df.select_dtypes(include=["float64", 'int64']).columns
    # df_numeric_summary = df[numeric_columns].describe()
    
    # plot the distributions to visualize outliers
    #for col in numeric_columns:
    #    df.hist(col)
    
    df = handle_outliers(df, method=OUTLIER_METHOD, manual_outlier_cols=MANUAL_OUTLIER_COLS)
    
    return df 


def impute_df(df, method='regression', max_iter=5, seed=1, n_neighbors=5, weights='uniform'):
    '''
    Impute missing column values using specified method. All columns are numeric, so we can use one method for all

    Parameters
    ----------
    df : DataFrame
    method : type of imputation to use, optional
        Options are 'mean', 'regression', and 'knn'. The default is 'regression'.
    max_iter : int, optional
        Max iterations for fitting regression model. The default is 5.
    seed : int, optional
        Random state. The default is 1.
    n_neighbors : int, optional
        Max neighbors to use for knn imputation. The default is 5.
    weights : string, optional
        Weight type to use for knn imputation. The default is 'uniform'.

    Returns
    -------
    df_transformed : DataFrame
    '''
        
    print(f"-- Performing imputation using {method} --")
    
    if method == 'mean':
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    elif method == 'regression':
        imputer = IterativeImputer(max_iter=max_iter, random_state=seed)
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    # TODO: stochastic regression imputation to introduce the noise that exists in reality for each feature
    #elif method == 'stochastic_regression':
    
    print('Fitting imputer to df')
    imputer.fit(df)
    cols = df.columns
    print('Imputing')
    df_transformed = pd.DataFrame(columns=cols, data=imputer.transform(df))
    
    return df_transformed


def preprocess(df, mixed_col_method='bool', outlier_method=OUTLIER_METHOD, manual_outlier_cols=MANUAL_OUTLIER_COLS, impute=True, impute_method=IMPUTE_METHOD):
    '''
    Preprocessing pipeline for data set

    Parameters
    ----------
    df : DataFrame
        input
    mixed_col_method : str, optional
        Which method to use for handling mixed dtype columns:
             'bool' - treat no as False, and yes and int values as True
             'exclude_yes' - remove 'yes' values and set 'no' to 0
             'exclude_string' - drop all string values
        The default is 'bool'.
    outlier_method: str, optional
        What to do with outliers ('drop' or 'cap'). The default is 'drop'.
    manual_outlier_cols : list, optional
        If specified, check only these cols for outliers. The default is ['Monthly rent payment']
    impute : bool, optional
        Whether or not to impute. The default is True.
    impute_method : str, optional
        Choice of 'mean', 'regression' or 'knn' imputation. The default is 'regression'.

    Returns
    -------
    df : DataFrame
        Processed dataframe
    '''
    
    # denote original index for traceability
    df['og_ix'] = df.index
    
    # set aside target and identifier columns
    nonprocessed_cols = ['og_ix', 'Id', 'Household level identifier', 'Target']
    df_non_processed = df[nonprocessed_cols]
    nonprocessed_cols.remove('og_ix')
    df.drop(columns=nonprocessed_cols, inplace=True)
    
    # remove redundant cols
    df = drop_redundant_cols(df)
    
    # remove columns and rows with large amounts of missing data 
    df = drop_mostly_missing(df)
    
    # boolean columns
    df = clean_boolean(df)
    
    # mixed columns
    df = clean_mixed_cols(df, method='bool')
    
    # numeric columns
    df = clean_numeric(df, outlier_method=OUTLIER_METHOD, manual_outlier_cols=MANUAL_OUTLIER_COLS)
    
    # impute missing values
    if impute:
        df = impute_df(df, method=IMPUTE_METHOD)
        
    # return non processed columns
    df = df.merge(df_non_processed, how='inner', on='og_ix')
    df.drop(columns=['og_ix'], inplace=True)
    
    return df

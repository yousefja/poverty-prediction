# -*- coding: utf-8 -*-
"""
This file is used to design and test new features for the model, which are 
formally implemented in feature_engineering.py

Created on Tue Jul 12 13:18:50 2022
"""

import numpy as np
import pandas as pd 


def create_calculated_features(df):
    '''
    Functionized version of the calculated features from feature_brainstorm.py
    Reference that file for any clarification on what the below features are.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    df : DataFrame
    '''
    
    df['calc_rent_per_person'] = df['Monthly rent payment'] / df['Total persons in the household']
    df['calc_rent_per_total_rooms'] = df['Monthly rent payment'] / df[' number of all rooms in the house']
    df['calc_rent_per_bedrooms'] = df['Monthly rent payment'] / df['number of bedrooms']
    df['calc_mobile_phones_per_person'] = df['# of mobile phones'] / df['Total persons in the household']
    df['calc_rooms_per_person'] = df[' number of all rooms in the house'] / df['Total persons in the household']
    df['calc_tablets_per_person'] = df['number of tablets household owns'] / df['Total persons in the household']
    df['calc_bedroom_to_room_ratio'] = df['number of bedrooms'] / df[' number of all rooms in the house']
    df['calc_ratio_adult_to_children_19'] = df['Number of adults in household'] / df['Number of children 0 to 19 in household']
    df['calc_ratio_adult_to_children_12'] = df['Number of adults in household'] / df['persons younger than 12 years of age']
    df['calc_ratio_under_12_to_total'] = df['persons younger than 12 years of age'] / df['Total persons in the household']
    df['calc_ratio_over_12_to_total'] = df['persons 12 years of age and older'] / df['Total persons in the household']
    df['calc_ratio_adult_to_total'] = df['Number of adults in household'] / df['Total persons in the household']
    df['calc_ratio_total_male_to_female'] = df['Total males in the household'] / df['Total females in the household']
    
    return df


def identify_incomplete_households(df):
    '''
    For each household, we know the number of residents, but this function 
    indicates wether or not we have data for each of these residents. If not, 
    the subsequent aggregate functions cannot be used since they will have 
    been calculated on incomplete household data.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    df_incomplete_households : DataFrame
        Household's (id's) with incomplete resident data
    '''
    
    # for each household, do I have all the data for each household member?
    num_households = df['Household level identifier'].nunique()                      # ensure all households accounted for
    df_household_count = df.groupby('Household level identifier')['Id'].agg('count') # get count of individual members for whom I have data
    df_household_count = df_household_count.reset_index()                            # convert to df
    df_household_count.rename(columns={'Id':'individual_count'}, inplace=True)       # rename column for clarity
    df_household_count = df_household_count.merge(df[['Household level identifier', 'Total persons in the household']], how='left', on='Household level identifier')
    df_household_count['have_all_individual_data'] = df_household_count['individual_count'] == df_household_count['Total persons in the household'] # see how manual count compares to count from dataset
    df_household_count.drop_duplicates(inplace=True)                                 # only need one row per household
    assert(len(df_household_count) == num_households) # ensure all households from original dataset are accounted for
    
    # For households where we don't have data for all the members, I'll simply replace the aggregate calculations with na after calculating agg features
    df_incomplete_households = df_household_count[df_household_count['have_all_individual_data'] == False]
    df_incomplete_households = pd.DataFrame(df_incomplete_households['Household level identifier'].drop_duplicates())
    df_incomplete_households['is_complete'] = False
    assert(len(df_incomplete_households) == (df_household_count['have_all_individual_data'].value_counts()[False])) # ensure all incomplete households accounted for 
    
    return df_incomplete_households    


def identify_head_of_house(df):
    '''
    For households with an indicated head of household, creates two one-hot 
    columns, household_head_is_male and household_head_is_female. Otherwise, these
    are simply null.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    df : DataFrame
    '''
    
    # indicate if head of house is male for each row
    df['household_head_is_male'] = df.apply(lambda x: x['=1 if household head'] and x['=1 if male'], axis=1) # first calculate for each row
    df['is_head_any'] = df.apply(lambda x: x['=1 if household head'], axis=1) # and also ensure we have a head of household at all
    
    # if true for any row in household, true for entire household
    df_head = df.groupby('Household level identifier').agg({'household_head_is_male': lambda x: x.any(),
                                                            'is_head_any': lambda x: x.any()})
    df.drop(columns=['household_head_is_male', 'is_head_any'], inplace=True) # these will be replaced by same col's in df_head, which are household-wide
    df_head = df_head.reset_index()
    df = df.merge(df_head, how='left', on='Household level identifier')
    df['household_head_is_female'] = ~df['household_head_is_male'] # and create similar column for female
    
    # for households where no one was the head, replace household_head_is_male/female with na
    df['household_head_is_male'] = df.apply(lambda x: x.household_head_is_male if x.is_head_any else np.nan, axis=1)
    df['household_head_is_female'] = df.apply(lambda x: x.household_head_is_female if x.is_head_any else np.nan, axis=1)
    
    return df
        

def calculate_school_aggregates(df):
    '''
    For adults (18+), calculates various features surrounding years of schooling

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    df_school_agg : DataFrame
        household-level aggregated features (one row per household id)
    '''
    
    df_adults = df[df['Age in years'] >= 18]
    df_school_agg = df_adults.groupby('Household level identifier').agg(household_min_schooling_adults_18 = ('years of schooling','min'),
                                                                  household_max_schooling_adults_18 = ('years of schooling','max'),
                                                                  household_avg_schooling_adults_18 = ('years of schooling','mean'))
    # FEATURE: household_schooling_range_adults_18
    # only applicable to multi-adult households (set to nan for all others)
    df_adult_count = df_adults['Household level identifier'].value_counts() # get number of adults per household
    df_adult_count = df_adult_count.reset_index()
    df_adult_count.columns = ['Household level identifier', 'adult_count']

    df_school_agg['household_schooling_range_adults_18'] = df_school_agg['household_max_schooling_adults_18'] - df_school_agg['household_min_schooling_adults_18']
    df_school_agg = df_school_agg.merge(df_adult_count, how='left', on='Household level identifier')
    df_school_agg['household_schooling_range_adults_18'] = np.where(df_school_agg.adult_count > 1,
                                                                    df_school_agg['household_schooling_range_adults_18'],
                                                                    np.nan)
    return df_school_agg


def calculate_age_aggregates(df):
    '''
    For all household members, calculates various features surrounding age

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    df_age_agg : DataFrame
        household-level aggregated features (one row per household id)
    '''
    
    df_age_agg = df.groupby('Household level identifier').agg(household_max_age = ('Age in years','max'),
                                                           household_min_age = ('Age in years','min'),
                                                           household_avg_age = ('Age in years','mean'),
                                                           household_med_age = ('Age in years','median'),
                                                           household_std_dev_age = ('Age in years', np.std))
    # FEATURE: household_age_range
    # only applicable to multi-person (all ages) households (set to nan for all others)
    df_person_count = df['Household level identifier'].value_counts() # get number of [all] members per household
    df_person_count = df_person_count.reset_index()
    df_person_count.columns = ['Household level identifier', 'person_count']

    df_age_agg['household_age_range'] = df_age_agg['household_max_age'] - df_age_agg['household_min_age']
    df_age_agg = df_age_agg.merge(df_person_count, how='left', on='Household level identifier')
    df_age_agg['household_age_range'] = np.where(df_age_agg.person_count > 1,
                                                 df_age_agg['household_age_range'],
                                                 np.nan)
    return df_age_agg


def append_aggregates(df, df_school_agg, df_age_agg, df_incomplete_households):
    '''
    Appends age and school aggregated features to the original dataframe,
    setting these features to np.nan if the given household does not have 
    data for all of it's residents

    Parameters
    ----------
    df : DataFrame
        Household dataframe
    df_school_agg : DataFrame
        From calculate_school_aggregates.
    df_age_agg : DataFrame
        From calculate_age_aggregates
    df_incomplete_households : DataFrame
        Households with incomplete member data.

    Returns
    -------
    df : DataFrame
        Household dataframe
    '''
    
    # all household level aggregate features 
    agg_columns = ['household_max_age',
                 'household_min_age',
                 'household_avg_age',
                 'household_std_dev_age',
                 'household_med_age',
                 'household_age_range',
                 'household_min_schooling_adults_18',
                 'household_max_schooling_adults_18',
                 'household_avg_schooling_adults_18',
                 'household_schooling_range_adults_18']

    # rejoin temp df's to the original df 
    df_agg = df_school_agg.merge(df_age_agg, how='left', on='Household level identifier')
    df_agg.drop(columns=['person_count', 'adult_count'], inplace=True)
    df = df.merge(df_agg, how='left', on='Household level identifier')
    
    # for any incomplete households (don't have all household member data), replace agg columns with nan
    df = df.merge(df_incomplete_households, how='left', on='Household level identifier')
    for col in agg_columns:
        df[col] = df.apply(lambda x: np.nan if x['is_complete']==False else x[col], axis=1)
    df.drop(columns=['is_complete'], inplace=True)
    
    return df


def condition_mapper(row, house_part):
    '''
    A helper function for create_house_condition_features, used to determine
    the condition of a given part of the house based on the corresponding
    one-hot-encoded column in the dataframe

    Parameters
    ----------
    row : row in df
        Corresponds to a person
    house_part : string
        walls, roof, or floor

    Returns
    -------
    float
        condition level based on ordinal categorical descriptor
    '''
    
    # provides modularity since all house part features have the same naming format
    column_name = '=1 if ' + house_part + ' are '
    
    if row[column_name + 'bad']:
        return 1
    elif row[column_name + 'regular']:
        return 2
    elif row[column_name + 'good']:
        return 3
    else:
        return np.nan
    

def create_house_condition_features(df):
    '''
    Converts one-hot-encoded features into a single, ordinal feature for 
    the condition of each respecive part of the house (walls, floor, roof).

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    df : DataFrame
    '''
    
    df['calc_wall_condition'] = df.apply(lambda x: condition_mapper(x, 'walls'), axis=1)
    df['calc_roof_condition'] = df.apply(lambda x: condition_mapper(x, 'roof'), axis=1)
    df['calc_floor_condition'] = df.apply(lambda x: condition_mapper(x, 'floor'), axis=1)
    
    return df


def engineer_features(df):
    '''
    Tie together all of the preceeding feature engineering steps

    Parameters
    ----------
    df : DataFrame
        Household data
        
    Returns
    -------
    df : DataFrame
        Household data with engineered features
    '''
    
    print("Creating calculated features")
    df = create_calculated_features(df)
    
    print("Identifying incomplete households")
    df_incomplete_households = identify_incomplete_households(df)
    
    print("Identifying head of household")
    df = identify_head_of_house(df)
    
    print("Calculating school aggregates")
    df_school_agg = calculate_school_aggregates(df)
    
    print("Calculating age aggregates")
    df_age_agg = calculate_age_aggregates(df)
    
    print("Appending aggregate features to df")
    df = append_aggregates(df, df_school_agg, df_age_agg, df_incomplete_households)
    
    print("Creating house condition features")
    df = create_house_condition_features(df)
    
    return df


# TODO: bin age aince I'll probably be creating two models, one on the individual level, and one on the household level

###############################
# AUTOMATED FEATURE ENGINEERING
###############################

# TODO: revisit this functionality later

# https://towardsdatascience.com/automated-feature-engineering-in-python-99baf11cc219
# Deep Feature Synthesis using FeatureTools

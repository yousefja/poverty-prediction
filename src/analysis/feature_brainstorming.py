# -*- coding: utf-8 -*-
"""
This file is used to design and test new features for the model, which are 
formally implemented in feature_engineering.py (as such, this file should be
used for clarification as it is written in a "scratch work" format)

Created on Tue Jul 12 13:18:50 2022
"""

import numpy as np
import pandas as pd 
from util import decode_columns, preprocess


# import data
df_train = pd.read_csv('../data/train.csv/train.csv')
df_codebook = pd.read_csv('../data/codebook.csv')

# decode columns 
df_train = decode_columns(df_train, df_codebook)

# clean data
df = preprocess(df_train, impute=False)

################
# FEATURE REVIEW
################

all_cols = ['Id',
 'Monthly rent payment',
 '=1 Overcrowding by bedrooms',
 ' number of all rooms in the house',
 '=1 Overcrowding by rooms',
 '=1 has toilet in the household',
 '=1 if the household has refrigerator',
 'owns a tablet',
 'number of tablets household owns',
 'Males younger than 12 years of age',
 'Males 12 years of age and older',
 'Total males in the household',
 'Females younger than 12 years of age',
 'Females 12 years of age and older',
 'Total females in the household',
 'persons younger than 12 years of age',
 'persons 12 years of age and older',
 'Total persons in the household',
 'size of the household',
 'TamViv',
 'years of schooling',
 'Years behind in school',
 'household size',
 '=1 if predominant material on the outside wall is block or brick',
 '=1 if predominant material on the outside wall is socket (wood, zinc or absbesto',
 '=1 if predominant material on the outside wall is prefabricated or cement',
 '=1 if predominant material on the outside wall is waste material',
 '=1 if predominant material on the outside wall is wood ',
 '=1 if predominant material on the outside wall is zink',
 '=1 if predominant material on the outside wall is natural fibers',
 '=1 if predominant material on the outside wall is other',
 '=1 if predominant material on the floor is mosaic, ceramic, terrazo',
 '=1 if predominant material on the floor is cement',
 '=1 if predominant material on the floor is other',
 '=1 if predominant material on the floor is  natural material',
 '=1 if no floor at the household',
 '=1 if predominant material on the floor is wood',
 '=1 if predominant material on the roof is metal foil or zink',
 '=1 if predominant material on the roof is fiber cement, mezzanine ',
 '=1 if predominant material on the roof is natural fibers',
 '=1 if predominant material on the roof is other',
 '=1 if the house has ceiling',
 '=1 if water provision inside the dwelling',
 '=1 if water provision outside the dwelling',
 '=1 if no water provision',
 '=1 electricity from CNFL, ICE, ESPH/JASEC',
 '=1 electricity from private plant',
 '=1 no electricity in the dwelling',
 '=1 electricity from cooperative',
 '=1 no toilet in the dwelling',
 '=1 toilet connected to sewer or cesspool',
 '=1 toilet connected to  septic tank',
 '=1 toilet connected to black hole or letrine',
 '=1 toilet connected to other system',
 '=1 no main source of energy used for cooking (no kitchen)',
 '=1 main source of energy used for cooking electricity',
 '=1 main source of energy used for cooking gas',
 '=1 main source of energy used for cooking wood charcoal',
 '=1 if rubbish disposal mainly by tanker truck',
 '=1 if rubbish disposal mainly by botan hollow or buried',
 '=1 if rubbish disposal mainly by burning',
 '=1 if rubbish disposal mainly by throwing in an unoccupied space',
 '=1 if rubbish disposal mainly by throwing in river, creek or sea',
 '=1 if rubbish disposal mainly other',
 '=1 if walls are bad',
 '=1 if walls are regular',
 '=1 if walls are good',
 '=1 if roof are bad',
 '=1 if roof are regular',
 '=1 if roof are good',
 '=1 if floor are bad',
 '=1 if floor are regular',
 '=1 if floor are good',
 '=1 if disable person',
 '=1 if male',
 '=1 if female',
 '=1 if less than 10 years old',
 '=1 if free or coupled uunion',
 '=1 if married',
 '=1 if divorced',
 '=1 if separated',
 '=1 if widow/er',
 '=1 if single',
 '=1 if household head',
 '=1 if spouse/partner',
 '=1 if son/doughter',
 '=1 if stepson/doughter',
 '=1 if son/doughter in law',
 '=1 if grandson/doughter',
 '=1 if mother/father',
 '=1 if father/mother in law',
 '=1 if brother/sister',
 '=1 if brother/sister in law',
 '=1 if other family member',
 '=1 if other non family member',
 'Household level identifier',
 'Number of children 0 to 19 in household',
 'Number of adults in household',
 '# of individuals 65+ in the household',
 '# of total individuals in the household',
 'Dependency rate',
 'years of education of male head of household',
 'years of education of female head of household',
 'average years of education for adults (18+)',
 '=1 no level of education',
 '=1 incomplete primary',
 '=1 complete primary',
 '=1 incomplete academic secondary level',
 '=1 complete academic secondary level',
 '=1 incomplete technical secondary level',
 '=1 complete technical secondary level',
 '=1 undergraduate and higher education',
 '=1 postgraduate higher education',
 'number of bedrooms',
 '# persons per room',
 '=1 own and fully paid house',
 '=1 own, paying in installments',
 '=1 rented',
 '=1 precarious',
 '=1 other(assigned, borrowed)',
 '=1 if the household has notebook or desktop computer',
 '=1 if the household has TV',
 '=1 if mobile phone',
 '# of mobile phones',
 '=1 region Central',
 '=1 region Chorotega',
 '=1 region PacÃƒÂ\xadfico central',
 '=1 region Brunca',
 '=1 region Huetar AtlÃƒÂ¡ntica',
 '=1 region Huetar Norte',
 '=1 zona urbana',
 '=2 zona rural',
 'Age in years',
 'escolari squared',
 'age squared',
 'hogar_total squared',
 'edjefe squared',
 'hogar_nin squared',
 'overcrowding squared',
 'dependency squared',
 'meaned squared',
 'Age squared',
 'Target']

# looking only at non bool columns for clarity
df_non_bool = df.select_dtypes(exclude='bool')

non_bool_cols = ['Monthly rent payment',
 ' number of all rooms in the house',
 'number of tablets household owns',
 'Males younger than 12 years of age',
 'Males 12 years of age and older',
 'Total males in the household',
 'Females younger than 12 years of age',
 'Females 12 years of age and older',
 'Total females in the household',
 'persons younger than 12 years of age',
 'persons 12 years of age and older',
 'Total persons in the household',
 'TamViv',
 'years of schooling',
 'Years behind in school',
 'household size',
 'Number of children 0 to 19 in household',
 'Number of adults in household',
 '# of individuals 65+ in the household',
 'average years of education for adults (18+)',
 'number of bedrooms',
 '# persons per room',
 '# of mobile phones',
 'Age in years',
 'Id',
 'Household level identifier',]

#####################
# CALCULATED FEATURES
#####################

# feature brainstorming
calculated_features = ['rent_per_person',
                     'rent_per_bedrooms',
                     'rent_per_total_rooms',
                     'mobile_phones_per_person', 
                     'rooms_per_person', # this differs from the existing col in the df, which is the number of bedrooms per person
                     'tablets_per_person',
                     'bedroom_to_room_ratio', # of total rooms, how many are for sleeping = total bed / total rooms
                     'ratio_adult_to_children_19',
                     'ratio_adult_to_children_12',
                     'ratio_under_12_to_total',
                     'ratio_over_12_to_total',
                     'ratio_adult_to_total',
                     'ratio_total_male_to_adult_female',
]

# prepend 'calc_' to each feature (mostly to later determine if any of these were actually useful during feat selection)
calculated_features = ['calc_' + feat for feat in calculated_features]

# feature calculations
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

############################
# HOUSEHOLD LEVEL AGGREGATES
############################

#These features are created by considering all members in a given household

# QUESTION: for each household, do I have all the data for each household member?
num_households = df['Household level identifier'].nunique()                      # ensure all households accounted for
df_household_count = df.groupby('Household level identifier')['Id'].agg('count') # get count of individual members for whom I have data
df_household_count = df_household_count.reset_index()                            # convert to df
df_household_count.rename(columns={'Id':'individual_count'}, inplace=True)       # rename column for clarity
df_household_count = df_household_count.merge(df[['Household level identifier', 'Total persons in the household']], how='left', on='Household level identifier')
df_household_count['have_all_individual_data'] = df_household_count['individual_count'] == df_household_count['Total persons in the household'] # see how manual count compares to count from dataset
df_household_count.drop_duplicates(inplace=True)                                 # only need one row per household
assert(len(df_household_count) == num_households) # ensure all households from original dataset are accounted for

# ANSWER: True: 2916, False: 47 -> we have the majority
df_household_count['have_all_individual_data'].value_counts()[False]

# For households where we don't have data for all the members, I'll simply replace the aggregate calculations with na after calculating agg features
df_incomplete_households = df_household_count[df_household_count['have_all_individual_data'] == False]
df_incomplete_households = pd.DataFrame(df_incomplete_households['Household level identifier'].drop_duplicates())
df_incomplete_households['is_complete'] = False
assert(len(df_incomplete_households) == (df_household_count['have_all_individual_data'].value_counts()[False])) # ensure all incomplete households accounted for 

# household level aggregate features 
agg_columns = ['max_age',
               'min_age',
               'avg_age',
               'std_dev_age',
               'med_age',
               'age_range', # diff bw youngest and oldest,
               'min_schooling_adults_18',
               'max_schooling_adults_18',
               'avg_schooling_adults_18',
               'schooling_range_adults_18',
               ]
# prepend 'household_' to each feature for traceability
agg_columns = ['household_' + feat for feat in agg_columns]

# -------------- FEATURES FOR ADULT (18+) HOUSEHOLD MEMBERS --------------
df_adults = df[df['Age in years'] >= 18]
df_tmp = df_adults.groupby('Household level identifier').agg(household_min_schooling_adults_18 = ('years of schooling','min'),
                                                              household_max_schooling_adults_18 = ('years of schooling','max'),
                                                              household_avg_schooling_adults_18 = ('years of schooling','mean'))
# FEATURE: household_schooling_range_adults_18
# only applicable to multi-adult households (set to nan for all others)
df_adult_count = df_adults['Household level identifier'].value_counts() # get number of adults per household
df_adult_count = df_adult_count.reset_index()
df_adult_count.columns = ['Household level identifier', 'adult_count']

df_tmp['household_schooling_range_adults_18'] = df_tmp['household_max_schooling_adults_18'] - df_tmp['household_min_schooling_adults_18']
df_tmp = df_tmp.merge(df_adult_count, how='left', on='Household level identifier')
df_tmp['household_schooling_range_adults_18'] = np.where(df_tmp.adult_count > 1,
                                                         df_tmp['household_schooling_range_adults_18'],
                                                         np.nan)

# -------------- FEATURES FOR ALL HOUSEHOLD MEMEBERS --------------
df_tmp2 = df.groupby('Household level identifier').agg(household_max_age = ('Age in years','max'),
                                                       household_min_age = ('Age in years','min'),
                                                       household_avg_age = ('Age in years','mean'),
                                                       household_med_age = ('Age in years','median'),
                                                       household_std_dev_age = ('Age in years', np.std))
# FEATURE: household_age_range
# only applicable to multi-person (all ages) households (set to nan for all others)
df_person_count = df['Household level identifier'].value_counts() # get number of [all] members per household
df_person_count = df_person_count.reset_index()
df_person_count.columns = ['Household level identifier', 'person_count']

df_tmp2['household_age_range'] = df_tmp2['household_max_age'] - df_tmp2['household_min_age']
df_tmp2 = df_tmp2.merge(df_person_count, how='left', on='Household level identifier')
df_tmp2['household_age_range'] = np.where(df_tmp2.person_count > 1,
                                          df_tmp2['household_age_range'],
                                          np.nan)

# rejoin temp df's to the original df 
df_tmp = df_tmp.merge(df_tmp2, how='left', on='Household level identifier')
df_tmp.drop(columns=['person_count', 'adult_count'], inplace=True)
df = df.merge(df_tmp, how='left', on='Household level identifier')

# for any incomplete households (don't have all household member data), replace agg columns with nan
df = df.merge(df_incomplete_households, how='left', on='Household level identifier')
for col in agg_columns:
    df[col] = df.apply(lambda x: np.nan if x['is_complete']==False else x[col], axis=1)
df.drop(columns=['is_complete'], inplace=True)

# FEATURE: household_head_is_male -- is household head m or f
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

#########
# BINNING
#########

# TODO: bin age aince I'll probably be creating two models, one on the individual level, and one on the household level

# can't necessarily tell if these values are ordinal (regular < good ?), so I'll try out binning, but keep the one-hot versions as well
'''
 '=1 if walls are bad',
 '=1 if walls are regular',
 '=1 if walls are good',
 '=1 if roof are bad',
 '=1 if roof are regular',
 '=1 if roof are good',
 '=1 if floor are bad',
 '=1 if floor are regular',
 '=1 if floor are good',
'''

def condition_mapper(row, house_part):
    
    column_name = '=1 if ' + house_part + ' are '
    
    if row[column_name + 'bad']:
        return 1
    elif row[column_name + 'regular']:
        return 2
    elif row[column_name + 'good']:
        return 3
    else:
        return np.nan
    
df['calc_wall_condition'] = df.apply(lambda x: condition_mapper(x, 'walls'), axis=1)
df['calc_roof_condition'] = df.apply(lambda x: condition_mapper(x, 'roof'), axis=1)
df['calc_floor_condition'] = df.apply(lambda x: condition_mapper(x, 'floor'), axis=1)
 
###############################
# AUTOMATED FEATURE ENGINEERING
###############################

# TODO: revisit this functionality later

# https://towardsdatascience.com/automated-feature-engineering-in-python-99baf11cc219
# Deep Feature Synthesis using FeatureTools


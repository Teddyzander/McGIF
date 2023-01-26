import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv('data/processed-data.csv')

# make races Cauc, Afr, Hisp, Other
data.loc[data.Ethnic_Code_Text == 'Native American', 'Ethnic_Code_Text'] = 'Other'
data.loc[data.Ethnic_Code_Text == 'Asian', 'Ethnic_Code_Text'] = 'Other'
data.loc[data.Ethnic_Code_Text == 'Arabic', 'Ethnic_Code_Text'] = 'Other'
data.loc[data.Ethnic_Code_Text == 'Oriental', 'Ethnic_Code_Text'] = 'Other'
data.loc[data.Ethnic_Code_Text == 'African-Am', 'Ethnic_Code_Text'] = 'African-American'
race_count = pd.DataFrame(data['Ethnic_Code_Text'].value_counts())
race_count.rename(columns={'Ethnic_Code_Text': 'totals'}, inplace=True)

# get total number of each race
race_count.transpose().to_csv('data/compas_totals.csv')

# split dataframes by race
data_afram = data[data['Ethnic_Code_Text'] == 'African-American']
data_cauc = data[data['Ethnic_Code_Text'] == 'Caucasian']
data_hisp = data[data['Ethnic_Code_Text'] == 'Hispanic']
data_other = data[data['Ethnic_Code_Text'] == 'Other']

#calculate cdf for scores
ranges = np.linspace(0, 100, 101)
cdf_afram = np.cumsum(data_afram['NormScore'].groupby(pd.cut(data_afram.NormScore, ranges)).count()) / \
            int(race_count.iloc[0])
cdf_cauc = np.cumsum(data_cauc['NormScore'].groupby(pd.cut(data_cauc.NormScore, ranges)).count()) /\
           int(race_count.iloc[1])
cdf_hisp = np.cumsum(data_hisp['NormScore'].groupby(pd.cut(data_hisp.NormScore, ranges)).count()) /\
           int(race_count.iloc[2])
cdf_other = np.cumsum(data_other['NormScore'].groupby(pd.cut(data_other.NormScore, ranges)).count()) /\
           int(race_count.iloc[3])
cdf = {'score': np.linspace(1, 100, 100),
       'African-American': cdf_afram.values,
       'Caucasian': cdf_cauc.values,
       'Hispanic': cdf_hisp.values,
       'Other': cdf_other.values}
cdf_totals = pd.DataFrame(data=cdf)
cdf_totals.to_csv('data/test.csv', index=False)

print('stop')

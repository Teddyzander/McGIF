import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# get raw data
raw_data = pd.read_csv('raw_data/compas-scores-raw.csv')
# Only keep Risk of Recidivism data
raw_data = raw_data[raw_data.DisplayText == 'Risk of Recidivism']
# only keep names, race, birthdays, and scores
raw_data = raw_data[['FirstName', 'LastName', 'Ethnic_Code_Text', 'DateOfBirth', 'RawScore']]
# correct date format
raw_data['DateOfBirth'] = pd.to_datetime(raw_data['DateOfBirth'], errors='ignore')
# make names lower case (for matching)
raw_data['FirstName'] = raw_data['FirstName'].str.lower()
raw_data['LastName'] = raw_data['LastName'].str.lower()
# normalise scores
raw_data['NormScore'] = np.round(((raw_data['RawScore'] - min(raw_data['RawScore'])) /
                                  (max(raw_data['RawScore']) - min(raw_data['RawScore']))) * 100, 0).astype('int')

# get target data
target_data = pd.read_csv('raw_data/compas-scores-two-years.csv')
# only keep names, birthdays, and scores
target_data = target_data[['first', 'last', 'dob', 'two_year_recid']]
# correct date format
target_data['dob'] = pd.to_datetime(target_data['dob'], errors='ignore')
# make names lower case (for matching)
target_data['first'] = target_data['first'].str.lower()
target_data['last'] = target_data['last'].str.lower()

# create a dataframe to hold good data
data = pd.DataFrame.copy(raw_data)
data['target'] = -1

# find matching values
for data_i, data_r in data.iterrows():
    for target_i, target_r in target_data.iterrows():
        if (data_r['FirstName'] == target_r['first'] and data_r['LastName'] == target_r['last'] and
                data_r['DateOfBirth'] == target_r['dob']):
            data.loc[data_i, 'target'] = int(target_r['two_year_recid'])

# remove rows with no target
data = data[data.target != -1]
# save data
data.to_csv('data/processed-data.csv')

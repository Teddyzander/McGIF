import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# fetch compas data set
compas_dataset_raw = pd.read_csv('../dataSorter/raw_data/compas-scores-raw.csv')
compas_dataset = pd.read_csv('../dataSorter/raw_data/compas-scores-two-years.csv')
compas_dataset = compas_dataset[
    (compas_dataset['race'] == 'Caucasian') | (compas_dataset['race'] == 'African-American')]

# format data for consistency
compas_dataset_raw = compas_dataset_raw.applymap(lambda s: s.lower() if type(s) == str else s)
compas_dataset = compas_dataset.applymap(lambda s: s.lower() if type(s) == str else s)
compas_dataset_raw['DateOfBirth'] = pd.to_datetime(compas_dataset_raw.DateOfBirth)
compas_dataset['dob'] = pd.to_datetime(compas_dataset.dob)


raw = np.zeros(len(compas_dataset))
for i in range(len(compas_dataset)):
    first_name = compas_dataset['first'].iloc[i]
    last_name = compas_dataset['last'].iloc[i]
    dob = compas_dataset['dob'].iloc[i]
    temp = compas_dataset_raw.loc[(compas_dataset_raw['FirstName'] == first_name) &
                                    (compas_dataset_raw['LastName'] == last_name) &
                                    (compas_dataset_raw['DateOfBirth'] == dob) &
                                    (compas_dataset_raw['DisplayText'] == 'risk of recidivism')]['RawScore']
    if len(temp) > 1:
        temp = temp.iloc[len(temp)-1]
    elif len(temp) == 0:
        temp = np.nan

    raw[i] = temp

# normalise scores
normalised_Score = np.round((raw - np.nanmin(raw)) / (np.nanmax(raw) - np.nanmin(raw)), 2) * 100
compas_dataset['normalised_Scores'] = normalised_Score
compas_dataset = compas_dataset[compas_dataset['normalised_Scores'].notna()]

# sort data set into race_sex group combinations
compas_dataset_Male = compas_dataset.loc[compas_dataset['sex'] == 'male']
compas_dataset_Female = compas_dataset.loc[compas_dataset['sex'] == 'female']
compas_dataset_W_Male = compas_dataset_Male.loc[compas_dataset_Male['race'] == 'caucasian']
compas_dataset_W_Female = compas_dataset_Female.loc[compas_dataset_Female['race'] == 'caucasian']
compas_dataset_AF_Male = compas_dataset_Male.loc[compas_dataset_Male['race'] == 'african-american']
compas_dataset_AF_Female = compas_dataset_Female.loc[compas_dataset_Female['race'] == 'african-american']

# store data in dict
data_dict = {0: compas_dataset_W_Male,
             1: compas_dataset_W_Female,
             2: compas_dataset_AF_Male,
             3: compas_dataset_AF_Female}

# get totals for each class and save to CSV
totals = [None, None, None, None, None]
totals[0] = 'Recidism Risk'
for i in range(1, 5):
    totals[i] = len(data_dict[i - 1])

totals_df = pd.DataFrame(totals).T
totals_df.columns = ['Kind', 'Caucasian Male', 'Caucasian Female',
                     'African-American Male', 'African-American Female']

totals_df.to_csv('../compas_totals.csv', index=False)

cdf = np.zeros((101, 5))
for i in range(0, 101):
    for j in range(0, 5):
        if j == 0:
            cdf[i, j] = i
        else:
            cdf[i, j] = (len(data_dict[j - 1].loc[data_dict[j - 1]['normalised_Scores'] <= i]) / totals[j] * 100)

cdf_df = pd.DataFrame(np.round(cdf, 2))
cdf_df.columns = ['Score', 'Caucasian Male', 'Caucasian Female',
                  'African-American Male', 'African-American Female']

cdf_df.to_csv('../compas_cdf.csv', index=False)

perf = np.zeros((101, 5))
for i in range(0, 101):
    for j in range(0, 5):
        if j == 0:
            perf[i, j] = i
        else:
            perf[i, j] = len(data_dict[j - 1][
                (data_dict[j - 1]['normalised_Scores'] == i) &
                (data_dict[j - 1]['two_year_recid'] == 1)]) / totals[j] * 100
perf_df = pd.DataFrame(np.round(perf, 2))
perf_df.columns = ['Score', 'Caucasian Male', 'Caucasian Female',
                  'African-American Male', 'African-American Female']

perf_df.to_csv('../compas_performance.csv', index=False)



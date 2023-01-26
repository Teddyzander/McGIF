import pandas
import pandas as pd
import numpy as np
from statsmodels.formula.api import logit

# following prorepublica analysis
dataURL = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years-violent.csv'
dfRaw = pd.read_csv(dataURL)

dfFiltered = (dfRaw[['age', 'c_charge_degree', 'race', 'age_cat', 'v_score_text',
             'sex', 'priors_count', 'days_b_screening_arrest', 'v_decile_score',
             'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]
             .loc[(dfRaw['days_b_screening_arrest'] <= 30) & (dfRaw['days_b_screening_arrest'] >= -30), :]
             .loc[dfRaw['is_recid'] != -1, :]
             .loc[dfRaw['c_charge_degree'] != 'O', :]
             .loc[dfRaw['v_score_text'] != 'N/A', :])

dfsens = (dfRaw['race'])

# run analysis
catCols = ['v_score_text','age_cat','race','sex','c_charge_degree']
dfFiltered.loc[:,catCols] = dfFiltered.loc[:,catCols].astype('category')
dfDummies = pd.get_dummies(data = dfFiltered, columns=catCols)
new_column_names = [col.lstrip().rstrip().lower().replace(" ", "_").replace("-", "_") for col in dfDummies.columns]
dfDummies.columns = new_column_names
# We want another variable that combines Medium and High
dfDummies['v_score_text_medhi'] = dfDummies['v_score_text_medium'] + dfDummies['v_score_text_high']

formula = 'v_score_text_medhi ~ sex_female + age_cat_greater_than_45 + age_cat_less_than_25 + race_african_american + race_asian + race_hispanic + race_native_american + race_other + priors_count + c_charge_degree_m + two_year_recid'

score_mod = logit(formula, data = dfDummies).fit()
print(score_mod.summary())

RAY = dfFiltered[['v_decile_score', 'race', 'is_recid']]
RAY = RAY.replace(['Asian'], 'Other')
RAY = RAY.replace(['Native American'], 'Other')
temp1 = RAY['race'].value_counts()
temp2 = RAY.groupby(["race", "v_decile_score"]).size()
temp3 = RAY.groupby(["race", "v_decile_score", "is_recid"]).size()

print('stop')
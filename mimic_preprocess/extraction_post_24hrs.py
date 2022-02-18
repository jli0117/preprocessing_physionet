import pickle
import pandas as pd
import numpy as np

# define params
MAX_DURATION = 24
ID_COL = 'icustay_id'

# load dataset
with open('output/df_interv_merged.pkl', 'rb') as fp:
    interv = pickle.load(fp)
interv = interv.set_index(['icustay_id', 'hours_in'])

DATA_FILEPATH = '../../code/preprocessing/MIMIC-EXTRACT-data-preprocessing/data/all_hourly_data.h5'
vital = pd.read_hdf(DATA_FILEPATH, 'vitals_labs')

idx = pd.IndexSlice
vital = vital.loc[:, idx[:, 'mean']]
vital = vital.droplevel(1, axis=1)
vital = vital.reset_index()
vital = vital.drop(['subject_id', 'hadm_id'], axis=1)
vital = vital.set_index(['icustay_id', 'hours_in'])

with open('output/data_statics_.pkl', 'rb') as fp:
    statics = pickle.load(fp)

with open('output/pid_list.pkl', 'rb') as fp:
    pid_list = pickle.load(fp)


"""
###################
Part1: cohort selection
###################
"""
statics = statics[statics.max_hours > MAX_DURATION]

vital = vital[vital.index.get_level_values('icustay_id').\
                isin(set(statics.index.get_level_values('icustay_id')))]

interv = interv[interv.index.get_level_values('icustay_id').\
                isin(set(statics.index.get_level_values('icustay_id')))]

vital_list = []
for icu_stay_id in vital.index.get_level_values(ID_COL).unique():
    vital_each_patient = vital.loc[pd.IndexSlice[icu_stay_id], :]
    max_hours_ = vital_each_patient.index.get_level_values("hours_in")[-1]
    assert (max_hours_+1) == len(vital_each_patient)
    hours_in_interval = list(range(int(max_hours_)-MAX_DURATION+1, int(max_hours_+1)))
    vital_each_patient_ = vital_each_patient.loc[pd.IndexSlice[hours_in_interval], :]
    vital_each_patient_ = vital_each_patient_.reset_index()
    vital_each_patient_[ID_COL] = icu_stay_id
    vital_each_patient_ = vital_each_patient_.set_index([ID_COL, "hours_in"])
    vital_list.append(vital_each_patient_)
vital_24hrs = pd.concat(vital_list)

interv_list = []
for icu_stay_id in interv.index.get_level_values(ID_COL).unique():
    interv_each_patient = interv.loc[pd.IndexSlice[icu_stay_id], :]
    max_hours_ = interv_each_patient.index.get_level_values("hours_in")[-1]
    assert (max_hours_+1) == len(interv_each_patient)
    hours_in_interval = list(range(int(max_hours_)-MAX_DURATION+1, int(max_hours_+1)))
    interv_each_patient_ = interv_each_patient.loc[pd.IndexSlice[hours_in_interval], :]
    interv_each_patient_ = interv_each_patient_.reset_index()
    interv_each_patient_[ID_COL] = icu_stay_id
    interv_each_patient_ = interv_each_patient_.set_index([ID_COL, "hours_in"])
    interv_list.append(interv_each_patient_)
interv_24hrs = pd.concat(interv_list)


interv_idx, vital_idx, statics_idx = [df.index.get_level_values('icustay_id') for df in (interv_24hrs, vital_24hrs, statics)]
assert set(interv_idx) == set(vital_idx), "icustay_id pools differ!"
assert set(vital_idx) == set(statics_idx), "icustay_id pools differ!"

print('Shape of vital signs : ', vital_24hrs.shape)
print('Shape of medical intervs : ', interv_24hrs.shape)
print('Shape of statics : ', statics.shape)

conditions = [ (statics['mort_icu'] == 1),
               (statics['mort_icu'] == 0) & (statics['mort_hosp'] == 1),
               (statics['mort_hosp'] == 0) & (statics['readmission_30'] == 1),
               (statics['readmission_30'] == 0) ]

values = [ 'mortality_icu', 'mortality_hos', 'read_30days', 'no_read_30days' ]
statics['conditions'] = np.select(conditions, values)
assert len(statics[statics['conditions'].isna()]) == 0

print(statics.conditions.value_counts())

"""
###################
Part2: imputation & normalisation
###################
"""
def simple_imputer_tr(df_out):
    df_out = df_out.dropna(axis=1, how='all')
    icustay_means = df_out.groupby(ID_COL).mean()
    global_means = df_out.mean(axis=0)
    df_out_fill = df_out.groupby(ID_COL).fillna(
        method='ffill').groupby(ID_COL).fillna(icustay_means).fillna(global_means)
    return df_out_fill

vital_24hrs_imp = simple_imputer_tr(vital_24hrs)

assert not vital_24hrs_imp.isnull().any().any()
assert not interv_24hrs.isnull().any().any()

def minmax(x):
    mins = x.min()
    maxes = x.max()
    x_std = (x - mins) / (maxes - mins)
    return x_std

vital_24hrs_std = vital_24hrs_imp.apply(lambda x: minmax(x))

"""
###################
Part3: convert into 3d matrix
###################
"""
def create_x_matrix(x):
    return x.iloc[:, 2:].values

vital_24hrs_3D = np.array(list(vital_24hrs_std.reset_index().groupby(ID_COL).apply(create_x_matrix)))
interv_24hrs_3D = np.array(list(interv_24hrs.reset_index().groupby(ID_COL).apply(create_x_matrix)))

print("vital_24hrs_3D tensor shape: ", vital_24hrs_3D.shape)
print("interv_24hrs_3D tensor shape: ", interv_24hrs_3D.shape)

with open('output/post24hrs/vital_sign_24hrs.pkl', 'wb') as outfile:
    pickle.dump(vital_24hrs_3D, outfile, pickle.HIGHEST_PROTOCOL)
with open('output/post24hrs/med_interv_24hrs.pkl', 'wb') as outfile:
    pickle.dump(interv_24hrs_3D, outfile, pickle.HIGHEST_PROTOCOL)

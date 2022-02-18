import pickle
import pandas as pd
import numpy as np
import os
from utils.utils import find_hourly_slots

# define params
MAX_DURATION = 24
ID_COL = 'patientunitstayid'

# load dataset
with open('output/patient/pats.pkl', 'rb') as fp:
    pats_hr = pickle.load(fp)

pats_hr = pats_hr.drop(['patienthealthsystemstayid', 'uniquepid'], axis=1)
data_statics = pats_hr.set_index(ID_COL)

with open('output/vital/vital_agg/vital_merge_all.pkl', 'rb') as fp:
    data_vital = pickle.load(fp)

with open('output/interv/medinterv.pkl', 'rb') as fp:
    data_interv = pickle.load(fp)

statics = data_statics[data_statics.unitdischargeoffset >= MAX_DURATION*60]

vital = data_vital[data_vital.index.get_level_values(ID_COL).\
                isin(set(statics.index.get_level_values(ID_COL)))]

interv = data_interv[data_interv.index.get_level_values(ID_COL).\
                isin(set(statics.index.get_level_values(ID_COL)))]

print("number of patients in patient: ", len(statics.index.get_level_values(ID_COL).unique()))
print("number of patients in vitalsigns: ", len(vital.index.get_level_values(ID_COL).unique()))
print("number of patients in interventions: ", len(interv.index.get_level_values(ID_COL).unique()))

vital_id = data_vital.index.get_level_values(ID_COL).unique().tolist()
statics = statics[statics.index.get_level_values(ID_COL).isin( vital_id )]
interv = interv[interv.index.get_level_values(ID_COL).isin( vital_id )]

print("number of patients in patient: ", len(statics.index.get_level_values(ID_COL).unique()))
print("number of patients in vitalsigns: ", len(vital.index.get_level_values(ID_COL).unique()))
print("number of patients in interventions: ", len(interv.index.get_level_values(ID_COL).unique()))

vital_list = []
for icu_stay_id in vital.index.get_level_values(ID_COL).unique():
    vital_each_patient = vital.loc[pd.IndexSlice[icu_stay_id], :]
    statics_each_patient = statics.loc[pd.IndexSlice[icu_stay_id], :]
    max_hours_ = find_hourly_slots(statics_each_patient.unitdischargeoffset)
    assert (max_hours_+1) == len(vital_each_patient)
    hours_in_interval = list(range(int(max_hours_)-MAX_DURATION+1, int(max_hours_+1)))
    vital_each_patient_ = vital_each_patient.loc[pd.IndexSlice[hours_in_interval], :]
    vital_each_patient_ = vital_each_patient_.reset_index()
    vital_each_patient_[ID_COL] = icu_stay_id
    vital_each_patient_ = vital_each_patient_.set_index([ID_COL, "itemoffset"])
    vital_list.append(vital_each_patient_)
vital_24hrs = pd.concat(vital_list)

interv_list = []
for icu_stay_id in interv.index.get_level_values(ID_COL).unique():
    interv_each_patient = interv.loc[pd.IndexSlice[icu_stay_id], :]
    statics_each_patient = statics.loc[pd.IndexSlice[icu_stay_id], :]
    max_hours_ = find_hourly_slots(statics_each_patient.unitdischargeoffset)
    assert max_hours_ == len(interv_each_patient)
    hours_in_interval = list(range(int(max_hours_)-MAX_DURATION, int(max_hours_)))
    interv_each_patient_ = interv_each_patient.loc[pd.IndexSlice[hours_in_interval], :]
    interv_each_patient_ = interv_each_patient_.reset_index()
    interv_each_patient_[ID_COL] = icu_stay_id
    interv_each_patient_ = interv_each_patient_.set_index([ID_COL, "timestamps"])
    interv_list.append(interv_each_patient_)
interv_24hrs = pd.concat(interv_list)

interv_idx, vital_idx, statics_idx = [df.index.get_level_values(ID_COL).unique() for df in (interv_24hrs, vital_24hrs, statics)]
statics_loc = statics.loc[vital_idx, :]
interv_24hrs_loc = interv_24hrs.loc[vital_idx, :]
interv_idx_loc, vital_idx, statics_idx_loc = [df.index.get_level_values(ID_COL) for df in (interv_24hrs_loc, vital_24hrs, statics_loc)]
assert set(interv_idx_loc) == set(vital_idx), "ICUSTAYID pools differ!"
assert set(vital_idx) == set(statics_idx_loc), "ICUSTAYID pools differ!"
print('Shape of vital signs : ', vital_24hrs.shape)     
print('Shape of medical intervs : ', interv_24hrs_loc.shape) 
print('Shape of statics : ', statics_loc.shape)    

interv_vaso = pd.Series(interv_24hrs[['dopamine', 'epinephrine', 'norepinephrine', 'phenylephrine', 'vasopressin',
                   'milrinone', 'dobutamine']].any(axis=1), name = "Vaso").to_frame()
interv_vaso.replace({False: 0, True: 1}, inplace=True)
interv_24hrs = pd.concat([interv_24hrs, interv_vaso], axis=1)


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

vital_24hrs_std = vital_24hrs_imp.copy()

col_mins = []
col_maxs = []
for column in vital_24hrs_std:
    col_mins.append(vital_24hrs_std[column].min())
    col_maxs.append(vital_24hrs_std[column].max())
vital_24hrs_std = vital_24hrs_std.apply(lambda x: minmax(x))
np.savez(os.path.join('output/data_24hrs', "min_max_vitalsigns.npz"), mins=col_mins, maxs=col_maxs, vital_24hrs_std=vital_24hrs_std)

def create_x_matrix(x): 
    return x.iloc[:, 2:].values

vital_24hrs_3D = np.array(list(vital_24hrs_std.reset_index().groupby('patientunitstayid').apply(create_x_matrix)))
interv_24hrs_3D = np.array(list(interv_24hrs.reset_index().groupby('patientunitstayid').apply(create_x_matrix)))

print("vital_24hrs_3D tensor shape: ", vital_24hrs_3D.shape)
print("interv_24hrs_3D tensor shape: ", interv_24hrs_3D.shape)


with open('output/data_24hrs/vital_sign_24hrs.pkl', 'wb') as outfile:
    pickle.dump(vital_24hrs_3D, outfile, pickle.HIGHEST_PROTOCOL)
with open('output/data_24hrs/med_interv_24hrs.pkl', 'wb') as outfile:
    pickle.dump(interv_24hrs_3D, outfile, pickle.HIGHEST_PROTOCOL)
    
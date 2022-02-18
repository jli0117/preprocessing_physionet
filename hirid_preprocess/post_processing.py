import pickle
import pandas as pd
import numpy as np
import glob

"""
###################
Part1: aggregate results based on all parquets
###################
"""
f_vital_list = [f for f in glob.glob("output/hourly_agg/data_vital_hourly-*.pkl")]
f_interv_list = [f for f in glob.glob("output/hourly_agg/data_interv_hourly-*.pkl")]

# merge all list from vital signs
data_vital_lst = []
for f in f_vital_list:
    with (open(f, "rb")) as file:
        vital_pkl = pickle.load(file)
    data_vital_lst.append(vital_pkl)
data_vital = pd.concat(data_vital_lst, axis=0)
print("the merged vital dataframe has the shape of {}".format(data_vital.shape))

# merge all list from pharma tables
data_interv_lst = []
for f in f_interv_list:
    with (open(f, "rb")) as file:
        interv_pkl = pickle.load(file)
    data_interv_lst.append(interv_pkl)
data_interv = pd.concat(data_interv_lst, axis=0)
print("the merged interv dataframe has the shape of {}".format(data_interv.shape))

with open('output/hourly_agg/data_vital_hourly_merged.pkl', 'wb') as outfile:
    pickle.dump(data_vital, outfile, pickle.HIGHEST_PROTOCOL)
with open('output/hourly_agg/data_interv_hourly_merged.pkl', 'wb') as outfile:
    pickle.dump(data_interv, outfile, pickle.HIGHEST_PROTOCOL)

"""
###################
Part2: select the 24 hours before icu discharge, select variables 
###################
"""

# define params
MAX_DURATION = 24
ID_COL = 'patientid'

# load dataset
with open('output/hourly_agg/data_vital_hourly_merged.pkl', 'rb') as fp:
    vital = pickle.load(fp)

with open('output/hourly_agg/data_interv_hourly_merged.pkl', 'rb') as fp:
    interv = pickle.load(fp)

vital_list = []
for icu_stay_id in vital.index.get_level_values(ID_COL).unique():
    vital_each_patient = vital.loc[pd.IndexSlice[icu_stay_id], :]
    max_hours_ = vital_each_patient.index.get_level_values("offset")[-1]
    assert (max_hours_+1) == len(vital_each_patient)
    hours_in_interval = list(range(int(max_hours_)-MAX_DURATION+1, int(max_hours_+1)))
    vital_each_patient_ = vital_each_patient.loc[pd.IndexSlice[hours_in_interval], :]
    vital_each_patient_ = vital_each_patient_.reset_index()
    vital_each_patient_[ID_COL] = icu_stay_id
    vital_each_patient_ = vital_each_patient_.set_index([ID_COL, "offset"])
    vital_list.append(vital_each_patient_)
vital_24hrs = pd.concat(vital_list)

interv_list = []
for icu_stay_id in interv.index.get_level_values(ID_COL).unique():
    interv_each_patient = interv.loc[pd.IndexSlice[icu_stay_id], :]
    max_hours_ = interv_each_patient.index.get_level_values("offset")[-1]
    assert (max_hours_+1) == len(interv_each_patient)
    hours_in_interval = list(range(int(max_hours_)-MAX_DURATION+1, int(max_hours_+1)))
    interv_each_patient_ = interv_each_patient.loc[pd.IndexSlice[hours_in_interval], :]
    interv_each_patient_ = interv_each_patient_.reset_index()
    interv_each_patient_[ID_COL] = icu_stay_id
    interv_each_patient_ = interv_each_patient_.set_index([ID_COL, "offset"])
    interv_list.append(interv_each_patient_)
interv_24hrs = pd.concat(interv_list)

interv_idx, vital_idx = [df.index.get_level_values(ID_COL).unique() for df in (interv_24hrs, vital_24hrs)]
assert set(interv_idx) == set(vital_idx), "ICUSTAYID pools differ!"
print("total number of patients are {}".format(len(interv_idx)))

with open('output/name_index/varname_varref_id.pkl', 'rb') as fp:
    varname_varref_id = pickle.load(fp)
with open('output/name_index/varname_pharma_id.pkl', 'rb') as fp:
    varname_pharma_id = pickle.load(fp)
with open('output/name_index/varname_varref_name.pkl', 'rb') as fp:
    varname_varref_nm = pickle.load(fp)
with open('output/name_index/varname_pharma_name.pkl', 'rb') as fp:
    varname_pharma_nm = pickle.load(fp)


vital_24hrs = vital_24hrs[varname_varref_id]
interv_24hrs = interv_24hrs[varname_pharma_id]
print(vital_24hrs.shape)
print(interv_24hrs.shape)

interv_vaso = pd.Series(interv_24hrs[ ['pm39', 'pm40', 'pm41', 'pm42', 'pm43', 'pm44', 'pm45', 'pm46'] ].any(axis=1), name = "Vaso").to_frame()
interv_vaso.replace({False: 0, True: 1}, inplace=True)
interv_24hrs = pd.concat([interv_24hrs, interv_vaso], axis=1)

varname_pharma_id.extend(["Vaso"])
varname_pharma_nm.extend(["Vaso"])


"""
###################
Part3: imputation
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

"""
###################
Part4: normalisation
###################
"""
def minmax(x):
    mins = x.min()
    maxes = x.max()
    x_std = (x - mins) / (maxes - mins)
    return x_std

vital_24hrs_std = vital_24hrs_imp.apply(lambda x: minmax(x))

"""
###################
Part5: convert into 3d matrix
###################
"""
with open('output/data_24hrs/vital_24hrs_std.pkl', 'rb') as fp:
    vital_24hrs_std = pickle.load(fp)

with open('output/data_24hrs/interv_24hrs.pkl', 'rb') as fp:
    interv_24hrs = pickle.load(fp)

def create_x_matrix(x):
    return x.iloc[:, 2:].values

vital_24hrs_3D = np.array(list(vital_24hrs_std.reset_index().groupby(ID_COL).apply(create_x_matrix)))
interv_24hrs_3D = np.array(list(interv_24hrs.reset_index().groupby(ID_COL).apply(create_x_matrix)))

with open('output/data_24hrs/vital_sign_24hrs.pkl', 'wb') as outfile:
    pickle.dump(vital_24hrs_3D, outfile, pickle.HIGHEST_PROTOCOL)
with open('output/data_24hrs/med_interv_24hrs.pkl', 'wb') as outfile:
    pickle.dump(interv_24hrs_3D, outfile, pickle.HIGHEST_PROTOCOL)

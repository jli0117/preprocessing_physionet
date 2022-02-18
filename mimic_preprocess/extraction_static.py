import pandas as pd
import numpy as np
import pickle
import os

# load data
DATA_FILEPATH = '../../code/preprocessing/MIMIC-EXTRACT-data-preprocessing/data/all_hourly_data.h5'
data_statics = pd.read_hdf(DATA_FILEPATH, 'patients')
print("number of patients: {}".format(len(data_statics)))

# cohort selection
data_statics_los = data_statics[data_statics.max_hours > 24]
print("number of patients: {}".format(len(data_statics_los)))

data_statics_los = data_statics_los[data_statics_los.max_hours < 10*24]
print("number of patients: {}".format(len(data_statics_los)))

data_statics_los_age = data_statics_los[(data_statics_los.age > 15)]
print("number of patients: {}".format(len(data_statics_los_age)))

data_statics_ = data_statics_los_age.reset_index()
assert len(data_statics_.icustay_id.unique()) == len(data_statics_)

# save the selected patient ids
pid_list = list(data_statics_.icustay_id.unique())
with open('output/pid_list.pkl', 'wb') as outfile:
    pickle.dump(pid_list, outfile, pickle.HIGHEST_PROTOCOL)

# save the selected table and patient id list
data_statics_ = data_statics_.set_index(['subject_id', 'hadm_id', 'icustay_id'])
with open('output/data_statics_.pkl', 'wb') as outfile:
    pickle.dump(data_statics_, outfile, pickle.HIGHEST_PROTOCOL)


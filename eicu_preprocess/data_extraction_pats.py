import os
import pickle
from utils.utils import dataframe_from_csv
from utils.pat_utils import filter_patients_on_age, filter_one_unit_stay, filter_patients_on_columns, \
    transform_gender, transform_ethnicity, transform_hospital_discharge_status, transform_unit_discharge_status, \
    transform_dx_into_id, filter_patients_on_columns_model, filter_max_hours, create_labels

eicu_path = '../../../eICU_data/eicu-collaborative-research-database-2.0'

print("===========> Processing patient chart <===========")
def read_patients_table(eicu_path):
    pats = dataframe_from_csv(os.path.join(eicu_path, 'patient.csv'), index_col=False)
    pats = filter_patients_on_age(pats, min_age=15, max_age=89)
    pats = filter_one_unit_stay(pats)
    pats = filter_max_hours(pats, max_hours=24, thres=240)
    pats = filter_patients_on_columns(pats)

    pats.update(transform_gender(pats.gender))
    pats.update(transform_ethnicity(pats.ethnicity))
    pats.update(transform_hospital_discharge_status(pats.hospitaldischargestatus))
    pats.update(transform_unit_discharge_status(pats.unitdischargestatus))
    pats = transform_dx_into_id(pats) 
    pats = filter_patients_on_columns_model(pats)
    pats = create_labels(pats)

    cohort = pats.patientunitstayid.unique()
    print("number of the cohort (unique patients): ", len(cohort))
    return pats, cohort
pats, cohort = read_patients_table(eicu_path)

with open('output/patient/pats.pkl', 'wb') as outfile:
    pickle.dump(pats, outfile, pickle.HIGHEST_PROTOCOL)
with open('output/patient/cohort.pkl', 'wb') as outfile:
    pickle.dump(cohort, outfile, pickle.HIGHEST_PROTOCOL)
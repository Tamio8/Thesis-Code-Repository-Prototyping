'''
Exploratory Extraction of Clinical Notes from MIMIC-III

This notebook explores clinical note availability and content for ICU patients in the MIMIC-III dataset.
The goal was to assess whether MIMIC-III could be used for the dynamic summary prototype.

While extracted notes were not ultimately used in the final prototype, this exploration informed decisions
about data availability design of the prototype. The process includes:

Steps followed:

1. Authenticate and connect to BigQuery with access to MIMIC-III tables.
2. Cohort definition: Adult septic ICU patients with first ICU admission and a stay >24h, using mimiciii_derived.icustay_detail and angus_sepsis.
3. Exclusion of immunocompromised patients using a wide set of ICD-9 codes based on Lu et al. (2021).
4. Filtering and exploration:
  Retrieved noteevents for septic, non-immunocompromised patients.
  Subsetted to patients with relevant note types (e.g., physician, nursing).
  Sampled younger patients (<=60 years) with ICU stays <=20 days.
  Calculated note lengths and number of notes per patient to assess documentation quality and density.
5. Exported CSVs/XLSX for preliminary inspection but did not use them in the final prototype design.

'''

# Install the latest version of the Google Cloud BigQuery client library
!pip install  --upgrade google-cloud-bigquery

from google.colab import auth # Import the Google Colab authentication module
# Trigger the user authentication flow
# This will prompt the user to authorize access to their Google account
auth.authenticate_user()
# Print confirmation message after successful authentication
print('Autheticated')

#Import libraries
import pandas as pd
from google.cloud import bigquery # Access data using Google BigQuery

# set google cloud project ID
project_id = 'big-cargo-439707-b2'

# create BigQuery Client
client = bigquery.Client(project=project_id)

#  This query defines the cohort_stay used for prototype 1: dynamic summary
#  Inclusion criteria:
#   adult: need to derive from
#   stay in ICU `physionet-data.mimiciv_icu`
#   In ICU for at least 24 hours
#   First ICU stay
#   septic: 'angus_sepsis'
%%bigquery df_cohortiii_stays_septic --project big-cargo-439707-b2
 
   SELECT subject_id, hadm_id, gender, dod, admittime, dischtime, los_hospital, admission_age, ethnicity_grouped, intime, outtime, los_icu
     FROM physionet-data.mimiciii_derived.icustay_detail
     WHERE admission_age >= 18.0 AND los_icu > 1.0 AND first_icu_stay = TRUE # age, length of stay, first icu stay
        AND subject_id IN (SELECT subject_id
                            FROM physionet-data.mimiciii_derived.angus_sepsis
                            WHERE angus_sepsis.angus = 1)

df_cohortiii_stays_septic.drop_duplicates()
df_cohortiii_stays_septic.shape[0] # number of patients: 15115

df_cohortiii_stays_septic.head()


#  This query defines the cohort_stay
 
#  Inclusion criteria:
#   adult: need to derive from
#   stay in ICU `physionet-data.mimiciv_icu`
#   In ICU for at least 24 hours
#   First ICU stay
%%bigquery df_notes --project big-cargo-439707-b2
 
   SELECT *
    FROM physionet-data.mimiciii_notes.noteevents


#check out what notes look like
#print(df_notes.TEXT[4])

# check categories
df_notes['CATEGORY'].unique()

# check description
d=df_notes['DESCRIPTION'].unique()
for i in d:
  print(i)

# notes from septic patients only

septic_iii = df_cohortiii_stays_septic['hadm_id'].tolist()

df_septic_iii_notes = df_notes[df_notes.HADM_ID.isin(septic_iii)]


# select immunocompromised patients
# immunocompromised patients as in Lu et al. (2021). Efficacy and safety of cortiocopsteroids for septic shock in immunocompromised patients: A cohort study from MIMIC
# here: immunocompromised as exclusion criteria
# ICD-9 Code
# HIV/AIDS -- 042-0449
# Lymphoma -- 20000-20238, 20250-20301, 2386, 2733,20302-20382
# Metastatic cancer -- 1960-1991, 20970-20975, 20979, 78951
# Solid-state tumors -- 1400-1729, 1740-1759, 179-1958, 20900-20924, 20925-2093, 20930-20936, 25801-25803
# Transplantation -- 99680-99689, 23877, 1992, V420-V427, V4281-V4284, V4289, V429
# Autoimmune disease -- 5550-5552, 35800-35801, 7100-7105, 7108-7109, 5565-5566, 5568-5569, 7010, 7140, 5559, 340, 7200, 6960, 35971, 4465
# NOTE here I go after the diagnoses_icd table, which is what the patient get's billed for after discharge. With that, I might miss immunocompromised patients that come into the ICU with that condition.
#BETTER: also go after prescriptions and check what medication a patient received, and maybe additionally, labevents
 
%%bigquery df_iii_immunecompromised --project big-cargo-439707-b2
   SELECT hadm_id
   FROM physionet-data.mimiciii_clinical.diagnoses_icd
   WHERE ICD9_CODE in ('042-0449', '20000-20238', '20250-20301', '2386', '2733','20302-20382', '1960-1991', '20970-20975', '20979', '78951', '1400-1729', '1740-1759',
   '179-1958', '20900-20924', '20925-2093', '20930-20936', '25801-25803', '99680-99689', '23877', '1992', 'V420-V427', 'V4281-V4284', 'V4289', 'V429',
   '5550-5552', '35800-35801', '7100-7105', '7108-7109', '5565-5566', '5568-5569', '7010', '7140', '5559', '340', '7200', '6960', '35971', '4465')

df_iii_immunecompromised.drop_duplicates()
df_iii_immunecompromised.shape[0] #there are 1517 immunocompromised patiens

df_cohortiii_stays_septic.info()

# exclude immunocompromised patients from df_cohortiii_stays_septic
immunecompromised = df_iii_immunecompromised['hadm_id'].tolist() #list of id with immunocompromised

df_cohortiii_stays_septic_immune = df_cohortiii_stays_septic[~df_cohortiii_stays_septic.hadm_id.isin(immunecompromised)]

df_cohortiii_stays_septic_immune.info() # new cohort has 14564 patients

# sort cohort via los
df_cohortiii_stays_septic_immune = df_cohortiii_stays_septic_immune.sort_values('los_icu')
df_cohortiii_stays_septic_immune

# just patients that have been there 20 days or less
df_cohortiii_stays_septic_immune_20 = df_cohortiii_stays_septic_immune[df_cohortiii_stays_septic_immune.los_icu <= 20]
df_cohortiii_stays_septic_immune_20.info() # that are still 13286

#select patients whose age is less than 60 from the df_cohortiii_stays_septic_immune_20 cohort
df_cohortiii_stays_septic_immune_20_age60 = df_cohortiii_stays_septic_immune_20[df_cohortiii_stays_septic_immune_20.admission_age <= 60]


# select randomly 150 patients
df_cohortiii_stays_septic_immune_20_age60_random150 = df_cohortiii_stays_septic_immune_20_age60.sample(n=150)
#sort via los
df_cohortiii_stays_septic_immune_20_age60_random150 = df_cohortiii_stays_septic_immune_20_age60_random150.sort_values('los_icu')

df_cohortiii_stays_septic_immune_20_age60_random150

# filter out notes that have too many missing notes from df_septic_iii_notes

# new df: subject_id, hadm_id, n_notes, and count for each category

df_notes_count = pd.DataFrame(df_septic_iii_notes.groupby('HADM_ID')['CATEGORY'].value_counts().unstack())

df_notes_count['n_notes'] = df_notes_count.sum(axis=1)

df_notes_count.sort_values(by=['n_notes'], ascending=False)

# take out all id that have no: general, nursing, nursing/other, physician
columns_to_check = ['Nursing', 'Nursing/other', 'Physician ']

df_notes_count_dropped = df_notes_count.dropna(subset=columns_to_check, how='all')

#make index a column
df_notes_count_dropped = df_notes_count_dropped.reset_index() #those are all notes

# filter df_notes_count_dropped id from df_septic_iii_notes
id_dropped = df_notes_count_dropped['HADM_ID'].to_list()

df_septic_iii_notes_dropped = df_septic_iii_notes[df_septic_iii_notes.HADM_ID.isin(id_dropped)]

# filter df_notes_count_dropped based on id in df_cohortiii_stays_septic_immune_20_age60_random150 = cohort

cohort = df_cohortiii_stays_septic_immune_20_age60_random150['hadm_id'].tolist() #list of id

df_septic_iii_cohort_notes = df_septic_iii_notes_dropped[df_septic_iii_notes.HADM_ID.isin(cohort)]

#check how many of those randomly selected id are in df_notes_count_dropped

df_septic_iii_cohort_notes['HADM_ID'].nunique() #110

# add a new column that is length of note
df_septic_iii_cohort_notes_length = df_septic_iii_cohort_notes.copy()
df_septic_iii_cohort_notes_length['text_length'] = df_septic_iii_cohort_notes_length['TEXT'].str.len()
#sort after text length
df_septic_iii_cohort_notes_length = df_septic_iii_cohort_notes_length.sort_values(by=['text_length'], ascending=False)
#add a column that shows the count of notes per patient id
df_septic_iii_cohort_notes_length['n_notes'] = df_septic_iii_cohort_notes_length.groupby('HADM_ID').HADM_ID.transform('count')

# merge df_notes_count_dropped and fitting id from df_cohortiii_stays_septic_immune_20_age60_random150 to have all info in one df
df_cohortiii_stays_septic_immune_20_age60_random150 = df_cohortiii_stays_septic_immune_20_age60_random150.rename(columns={'hadm_id': 'HADM_ID'})
df_notes_count_dropped_info = df_notes_count_dropped.merge(df_cohortiii_stays_septic_immune_20_age60_random150, on='HADM_ID', how='left')

#safe datasets as csv
df_cohortiii_stays_septic_immune_20_age60_random150.to_csv('df_cohortiii_stays_septic_immune_20_age60_random150') # used as: cohort
df_septic_iii_cohort_notes_length.to_csv('df_septic_iii_cohort_notes')# NOTES TO USE
df_septic_iii_cohort_notes_length.to_excel('x_df_septic_iii_cohort_notes.xlsx')# NOTES TO USE

df_notes_count_dropped.to_csv('df_notes_count')
#df_notes_count_dropped.to_excel('x_df_notes_count_dropped')
df_notes_count_dropped_info.to_csv('df_notes_count_dropped_info')
#df_notes_count_dropped_info.to_excel('x_df_notes_count_dropped_info')

# add a new column that is length of note
df_septic_iii_immune_notes_length = df_septic_iii_immune_notes.copy()
df_septic_iii_immune_notes_length['text_length'] = df_septic_iii_immune_notes_length['TEXT'].str.len()

df_septic_iii_immune_notes_length.describe()

# sort after longest note

df_septic_iii_immune_notes_length = df_septic_iii_immune_notes_length.sort_values(by=['text_length'], ascending=False)

df_septic_iii_immune_notes_length.head()

df_septic_iii_immune_notes_length['CATEGORY'].unique()

#count how many notes per patient

df_septic_iii_immune_notes_length['n_notes'] = df_septic_iii_immune_notes_length.groupby('SUBJECT_ID').SUBJECT_ID.transform('count')
df_septic_iii_immune_notes_length

# new df: subject_id, n_notes, and count for each category

df_notes_count = pd.DataFrame(df_septic_iii_immune_notes_length.groupby('SUBJECT_ID')['CATEGORY'].value_counts().unstack())

df_notes_count['n_notes'] = df_notes_count.sum(axis=1)

df_notes_count.sort_values(by=['n_notes'], ascending=False)

df_notes_count.info()

df_notes_count.to_excel('df_notes_count.xlsx')

# saving
df_septic_iii_immune_notes_length.to_excel('df_septic_iii_immune_notes_length.xlsx')

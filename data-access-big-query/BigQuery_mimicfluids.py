'''
This notebook documents the step-by-step filtering process used to define a cohort of
ICU patients from the MIMIC-IV dataset, for use in a fluid balance tracker prototype.

The primary dataset source is the MIMIC-IV database, accessed via Google BigQuery in Google Colab.
The process includes authentication, cohort definition, and export of relevant patient information (demographics, medications, vitals, diagnoses, and discharge notes).

Steps followed:
1. Authenticate and connect to BigQuery using project ID big-cargo-439707-b2.
2. Initial cohort: Adult ICU patients with >= 24h ICU stay and a documented diagnosis of sepsis-3.
3. Clinical Filtering:
    Include only patients with invasive ventilation.
    Further include only patients with invasive lines (IV).
    Exclude immunocompromised patients based on ICD codes.
    Include only those with a diagnosis of burn injury.
4. Event Extraction:
    Retrieve and export relevant fluid-related events from inputevents, outputevents, and ingredientevents.
    Supplement with additional ICU data: chartevents, datetimeevents, procedureevents, and item definitions
    from d_items.

Output:
CSV exports of filtered cohorts and event tables, used for local analysis and prototyping.
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

#  This query defines the cohort_stay used for prototype 3 (same as for prototype 2)
 
#  Inclusion criteria:
#   adult `physionet-data.mimiciv_hosp.patients`
#   stay in ICU `physionet-data.mimiciv_icu`
#   In ICU for at least 24 hours
#   First ICU admission
 
%%bigquery df_cohort_stays_septic --project big-cargo-439707-b2
 
   SELECT subject_id, stay_id, gender, dod, admittime, dischtime, los_hospital, admission_age, race, icu_intime, icu_outtime, los_icu
     FROM physionet-data.mimiciv_derived.icustay_detail
     WHERE admission_age >= 18.0 AND los_icu > 1.0 AND first_icu_stay = TRUE # age, length of stay, first icu stay
        AND subject_id IN (SELECT subject_id
                            FROM physionet-data.mimiciv_derived.sepsis3
                            WHERE sepsis3.sepsis3 = TRUE) # sepsis 3 true

df_cohort_stays_septic.drop_duplicates()
df_cohort_stays_septic.shape[0] #number of patients in cohort_stay alone without sepsis is 51376

"""Select patients that are ventilated and have an IV"""

# select patients that are ventilated: mimiciv_derived.ventilation from cohort 104 in similarity
# subject_id IN (10148533, 15479218, 13804231, 15383083, 18614648, 14635008, 18877846, 19908221, 11233081, 15694968, 16044793, 13267235, 17589058, 14120635, 18888515, 18172425, 17524724, 19832679, 15248985, 14603776, 16076363, 19509653, 17170716, 15526304, 15752366, 15716202, 15579080, 17760909, 19970265, 13179092, 14645595, 12627644, 11166681, 19906564, 14581489, 12798063, 17760270, 11480283, 14098557, 14482934, 13869899, 15130525, 16103537, 19387278, 12710598, 11258077, 12798506, 18212558, 14198265, 12043836, 19482563, 10992347, 16952444, 19348626, 15299779, 14435499, 17209535, 13146468, 14275115, 17310431, 17447711, 12802332, 13549234, 11294917, 11900074, 14834230, 13050394, 13873595, 11376915, 10598816, 10935252, 11959467, 19311178, 14717765, 11536174, 13558665, 16285428, 13505628, 15703508, 14635022, 10644961, 18870809, 16517380, 16099802, 10933609, 17233716, 12621159, 10230202, 17792869, 17823750, 11425766, 18227829, 14976423, 16529186, 14280430, 13892963, 18629932, 15524760, 17546877, 14250712, 15712741, 14040144, 19043108, 10599715)
# stay_id IN (38372401, 38486717, 31885114, 30602234, 31847457, 39754636, 30698001, 39198142, 30162054, 33907802, 36837185, 35146959, 30782843, 31976189, 31047765, 33757419, 37189090, 39449136, 33271298, 32319147, 34564898, 38315255, 33832160, 37609236, 30012063, 33919649, 36279134, 39711363, 38356273, 38671812, 37766634, 36148917, 38192611, 30045760, 32452859, 36663037, 31557612, 32313856, 39743745, 39768103, 36780485, 32031913, 31588174, 34796801, 36299350, 38482726, 33439987, 36506153, 36552784, 31985637, 31676535, 32389325, 37787511, 30774936, 30626122, 35776095, 30696453, 35255147, 37744846, 37047409, 37000271, 38508770, 32620190, 36901358, 33035558, 39150944, 36805048, 32368927, 38874499, 38658006, 34473161, 36974556, 35580228, 38330612, 34818547, 30637692, 36557642, 31983448, 31041981, 38227266, 37259563, 32404823, 33882509, 35881882, 33485562, 32941531, 30624987, 31578659, 34634419, 34253386, 31185496, 39672431, 33187188, 37678620, 37171670, 34132590, 32416606, 31164656, 30006565, 37612264, 38536869, 35709738, 34607754, 36356955, 31986636)

%%bigquery df_cohort_ventilated --project big-cargo-439707-b2
 
   SELECT *
     FROM physionet-data.mimiciv_derived.ventilation
     WHERE stay_id IN (38372401, 38486717, 31885114, 30602234, 31847457, 39754636, 30698001, 39198142, 30162054, 33907802, 36837185, 35146959, 30782843, 31976189, 31047765, 33757419, 37189090, 39449136, 33271298, 32319147, 34564898, 38315255, 33832160, 37609236, 30012063, 33919649, 36279134, 39711363, 38356273, 38671812, 37766634, 36148917, 38192611, 30045760, 32452859, 36663037, 31557612, 32313856, 39743745, 39768103, 36780485, 32031913, 31588174, 34796801, 36299350, 38482726, 33439987, 36506153, 36552784, 31985637, 31676535, 32389325, 37787511, 30774936, 30626122, 35776095, 30696453, 35255147, 37744846, 37047409, 37000271, 38508770, 32620190, 36901358, 33035558, 39150944, 36805048, 32368927, 38874499, 38658006, 34473161, 36974556, 35580228, 38330612, 34818547, 30637692, 36557642, 31983448, 31041981, 38227266, 37259563, 32404823, 33882509, 35881882, 33485562, 32941531, 30624987, 31578659, 34634419, 34253386, 31185496, 39672431, 33187188, 37678620, 37171670, 34132590, 32416606, 31164656, 30006565, 37612264, 38536869, 35709738, 34607754, 36356955, 31986636)
     AND ventilation_status = 'InvasiveVent'

stay_id_vent = df_cohort_ventilated.stay_id.unique().tolist()
print(stay_id_vent)

# from those who are in the cohort AND are ventilated, check the ID that also have an IV
# select patients with an IV: mimiciv_derived.invasive_line
%%bigquery df_cohort_IV --project big-cargo-439707-b2
   SELECT *
     FROM physionet-data.mimiciv_derived.invasive_line
     WHERE stay_id IN (32313856, 34796801, 30696453, 30698001, 37609236, 32368927, 31847457, 31578659, 30006565, 33035558, 39768103, 39449136, 38356273, 38372401, 31885114, 36837185, 38227266, 35580228, 36557642, 36552784, 31047765, 36299350, 32404823, 33907802, 36356955, 32416606, 31676535, 30637692, 38874499, 30162054, 34607754, 33882509, 30774936, 35881882, 33919649, 32031913, 32319147, 36148917, 36805048, 38486717, 31041981, 39198142, 38671812, 36780485, 32389325, 37171670, 32941531, 36974556, 33832160, 37189090, 38508770, 31985637, 37766634, 33757419, 31557612, 36901358, 31164656, 33439987, 34818547, 38315255, 33485562, 32452859, 36663037, 31976189)

# id of cohort + ventilated + IV

stay_id_vent_iv = df_cohort_ventilated.stay_id.unique().tolist()
print(stay_id_vent_iv)
print(len(stay_id_vent_iv))

# get input and output events of those in stay_id_vent_iv, replace id with name and safe as csv
#get output events
%%bigquery df_cohort_output --project big-cargo-439707-b2
   SELECT *
   FROM physionet-data.mimiciv_icu.outputevents
   WHERE stay_id IN (32313856, 34796801, 30696453, 30698001, 37609236, 32368927, 31847457, 31578659, 30006565, 33035558, 39768103, 39449136, 38356273, 38372401, 31885114, 36837185, 38227266, 35580228, 36557642, 36552784, 31047765, 36299350, 32404823, 33907802, 36356955, 32416606, 31676535, 30637692, 38874499, 30162054, 34607754, 33882509, 30774936, 35881882, 33919649, 32031913, 32319147, 36148917, 36805048, 38486717, 31041981, 39198142, 38671812, 36780485, 32389325, 37171670, 32941531, 36974556, 33832160, 37189090, 38508770, 31985637, 37766634, 33757419, 31557612, 36901358, 31164656, 33439987, 34818547, 38315255, 33485562, 32452859, 36663037, 31976189)

#get input events events
%%bigquery df_cohort_inputevents --project big-cargo-439707-b2
   SELECT *
   FROM physionet-data.mimiciv_icu.inputevents
   WHERE stay_id IN (32313856, 34796801, 30696453, 30698001, 37609236, 32368927, 31847457, 31578659, 30006565, 33035558, 39768103, 39449136, 38356273, 38372401, 31885114, 36837185, 38227266, 35580228, 36557642, 36552784, 31047765, 36299350, 32404823, 33907802, 36356955, 32416606, 31676535, 30637692, 38874499, 30162054, 34607754, 33882509, 30774936, 35881882, 33919649, 32031913, 32319147, 36148917, 36805048, 38486717, 31041981, 39198142, 38671812, 36780485, 32389325, 37171670, 32941531, 36974556, 33832160, 37189090, 38508770, 31985637, 37766634, 33757419, 31557612, 36901358, 31164656, 33439987, 34818547, 38315255, 33485562, 32452859, 36663037, 31976189)

#get ingredient events events
%%bigquery df_cohort_ingredientevents --project big-cargo-439707-b2
   SELECT *
   FROM physionet-data.mimiciv_icu.ingredientevents
   WHERE stay_id IN (32313856, 34796801, 30696453, 30698001, 37609236, 32368927, 31847457, 31578659, 30006565, 33035558, 39768103, 39449136, 38356273, 38372401, 31885114, 36837185, 38227266, 35580228, 36557642, 36552784, 31047765, 36299350, 32404823, 33907802, 36356955, 32416606, 31676535, 30637692, 38874499, 30162054, 34607754, 33882509, 30774936, 35881882, 33919649, 32031913, 32319147, 36148917, 36805048, 38486717, 31041981, 39198142, 38671812, 36780485, 32389325, 37171670, 32941531, 36974556, 33832160, 37189090, 38508770, 31985637, 37766634, 33757419, 31557612, 36901358, 31164656, 33439987, 34818547, 38315255, 33485562, 32452859, 36663037, 31976189)

#add information from d_items
%%bigquery df_d_items --project big-cargo-439707-b2
   SELECT *
   FROM physionet-data.mimiciv_icu.d_items

#get procedure events from stay_id cohort in similarity 104
%%bigquery df_cohort_procedureevents --project big-cargo-439707-b2
   SELECT *
   FROM physionet-data.mimiciv_icu.procedureevents
   WHERE stay_id IN (38372401, 38486717, 31885114, 30602234, 31847457, 39754636, 30698001, 39198142, 30162054, 33907802, 36837185, 35146959, 30782843, 31976189, 31047765, 33757419, 37189090, 39449136, 33271298, 32319147, 34564898, 38315255, 33832160, 37609236, 30012063, 33919649, 36279134, 39711363, 38356273, 38671812, 37766634, 36148917, 38192611, 30045760, 32452859, 36663037, 31557612, 32313856, 39743745, 39768103, 36780485, 32031913, 31588174, 34796801, 36299350, 38482726, 33439987, 36506153, 36552784, 31985637, 31676535, 32389325, 37787511, 30774936, 30626122, 35776095, 30696453, 35255147, 37744846, 37047409, 37000271, 38508770, 32620190, 36901358, 33035558, 39150944, 36805048, 32368927, 38874499, 38658006, 34473161, 36974556, 35580228, 38330612, 34818547, 30637692, 36557642, 31983448, 31041981, 38227266, 37259563, 32404823, 33882509, 35881882, 33485562, 32941531, 30624987, 31578659, 34634419, 34253386, 31185496, 39672431, 33187188, 37678620, 37171670, 34132590, 32416606, 31164656, 30006565, 37612264, 38536869, 35709738, 34607754, 36356955, 31986636)


#get procedure events from stay_id cohort in similarity 104
%%bigquery df_cohort_datetimeevents --project big-cargo-439707-b2
   SELECT *
   FROM physionet-data.mimiciv_icu.datetimeevents
   WHERE stay_id IN (38372401, 38486717, 31885114, 30602234, 31847457, 39754636, 30698001, 39198142, 30162054, 33907802, 36837185, 35146959, 30782843, 31976189, 31047765, 33757419, 37189090, 39449136, 33271298, 32319147, 34564898, 38315255, 33832160, 37609236, 30012063, 33919649, 36279134, 39711363, 38356273, 38671812, 37766634, 36148917, 38192611, 30045760, 32452859, 36663037, 31557612, 32313856, 39743745, 39768103, 36780485, 32031913, 31588174, 34796801, 36299350, 38482726, 33439987, 36506153, 36552784, 31985637, 31676535, 32389325, 37787511, 30774936, 30626122, 35776095, 30696453, 35255147, 37744846, 37047409, 37000271, 38508770, 32620190, 36901358, 33035558, 39150944, 36805048, 32368927, 38874499, 38658006, 34473161, 36974556, 35580228, 38330612, 34818547, 30637692, 36557642, 31983448, 31041981, 38227266, 37259563, 32404823, 33882509, 35881882, 33485562, 32941531, 30624987, 31578659, 34634419, 34253386, 31185496, 39672431, 33187188, 37678620, 37171670, 34132590, 32416606, 31164656, 30006565, 37612264, 38536869, 35709738, 34607754, 36356955, 31986636)

#get procedure events from stay_id cohort in similarity 104
%%bigquery df_cohort_chartevents --project big-cargo-439707-b2
   SELECT *
   FROM physionet-data.mimiciv_icu.chartevents
   WHERE stay_id IN (38372401, 38486717, 31885114, 30602234, 31847457, 39754636, 30698001, 39198142, 30162054, 33907802, 36837185, 35146959, 30782843, 31976189, 31047765, 33757419, 37189090, 39449136, 33271298, 32319147, 34564898, 38315255, 33832160, 37609236, 30012063, 33919649, 36279134, 39711363, 38356273, 38671812, 37766634, 36148917, 38192611, 30045760, 32452859, 36663037, 31557612, 32313856, 39743745, 39768103, 36780485, 32031913, 31588174, 34796801, 36299350, 38482726, 33439987, 36506153, 36552784, 31985637, 31676535, 32389325, 37787511, 30774936, 30626122, 35776095, 30696453, 35255147, 37744846, 37047409, 37000271, 38508770, 32620190, 36901358, 33035558, 39150944, 36805048, 32368927, 38874499, 38658006, 34473161, 36974556, 35580228, 38330612, 34818547, 30637692, 36557642, 31983448, 31041981, 38227266, 37259563, 32404823, 33882509, 35881882, 33485562, 32941531, 30624987, 31578659, 34634419, 34253386, 31185496, 39672431, 33187188, 37678620, 37171670, 34132590, 32416606, 31164656, 30006565, 37612264, 38536869, 35709738, 34607754, 36356955, 31986636)

# safe those tables as csv

# safe inputevents of ventilated and iv as csv
df_cohort_inputevents.to_csv('df_cohort_vent_iv_inputevents')

# safe outputevents of ventilated and iv as csv
df_cohort_output.to_csv('df_cohort_vent_iv_outputevents')

# safe ingredientevents of ventilated and iv as csv
df_cohort_ingredientevents.to_csv('df_cohort_vent_iv_ingredientevents')

# safe d_items as csv
df_d_items.to_csv('df_d_items')

# safe procedureevents
df_cohort_procedureevents.to_csv('df_cohort_procedureevents')

# safe datetimeevents
df_cohort_datetimeevents.to_csv('df_cohort_datetimeevents')

# safe chartevents
df_cohort_chartevents.to_csv('df_cohort_chartevents')

df_cohort_ventilated['ventilation_status'].unique()
# 'None', 'HFNC', 'InvasiveVent', 'Tracheostomy', 'NonInvasiveVent', 'SupplementalOxygen'

count_InvasiveVent = df_cohort_ventilated['ventilation_status'].value_counts().get('InvasiveVent', 0)
print("Occurrences of 'InvasiveVent':", count_InvasiveVent)

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
%%bigquery df_cohort_immunecompromised --project big-cargo-439707-b2
   SELECT subject_id
   FROM physionet-data.mimiciv_hosp.diagnoses_icd
   WHERE icd_code in ('042-0449', '20000-20238', '20250-20301', '2386', '2733','20302-20382', '1960-1991', '20970-20975', '20979', '78951', '1400-1729', '1740-1759',
   '179-1958', '20900-20924', '20925-2093', '20930-20936', '25801-25803', '99680-99689', '23877', '1992', 'V420-V427', 'V4281-V4284', 'V4289', 'V429',
   '5550-5552', '35800-35801', '7100-7105', '7108-7109', '5565-5566', '5568-5569', '7010', '7140', '5559', '340', '7200', '6960', '35971', '4465')

df_cohort_immunecompromised.drop_duplicates()
df_cohort_immunecompromised.shape[0] #there are 9554 immunocompromised patiens

# exclude immunocompromised patients from cohort_stays_septic

immunecompromised = df_cohort_immunecompromised['subject_id'].tolist()

df_cohort_stays_septic_immune = df_cohort_stays_septic[~df_cohort_stays_septic.subject_id.isin(immunecompromised)]


#find ICD code for burns
%%bigquery df_icd_burns --project big-cargo-439707-b2
 
   SELECT * FROM physionet-data.mimiciv_hosp.d_icd_diagnoses
   WHERE long_title LIKE '%burn%'

# there are 334 diagnosis that include 'burn', some might not be relevant like maybe sunburn, depending on degree
burns_icd = df_icd_burns['icd_code'].tolist()

# check how many of those patients have a burns diagnosis
%%bigquery df_cohort_burns --project big-cargo-439707-b2
   SELECT subject_id
   FROM physionet-data.mimiciv_hosp.diagnoses_icd
   WHERE icd_code in ('69271', '69276', '69277', '7871', '9065', '9066', '9067', '9068', '9069', '9100', '9101', '9110', '9111', '9120', '9121', '9130', '9131', '9140', '9141', '9150', '9151', '9160', '9161', '9170', '9171', '9190', '9191', '9400', '9401', '9402', '9403', '9404', '9409', '94800', '94810', '94811', '94820', '94821', '94822', '94830', '94831', '94832', '94833', '94840', '94841', '94842', '94843', '94844', '94850', '94851', '94852', '94853', '94854', '94855', '94860', '94861', '94862', '94863', '94864', '94865', '94866', '94870', '94871', '94872', '94873', '94874', '94875', '94876', '94877', '94880', '94881', '94882', '94883', '94884', '94885', '94886', '94887', '94888', '94890', '94891', '94892', '94893', '94894', '94895', '94896', '94897', '94898', '94899', 'E8030', 'E8031', 'E8032', 'E8033', 'E8038', 'E8039', 'E8370', 'E8371', 'E8372', 'E8373', 'E8374', 'E8375', 'E8376', 'E8377', 'E8378', 'E8379', 'E8980', 'E8981', 'E9581', 'E9881', 'L55', 'L550', 'L551', 'L552', 'L559', 'M613', 'M6130', 'M6131', 'M61311', 'M61312', 'M61319', 'M6132', 'M61321', 'M61322', 'M61329', 'M6133', 'M61331', 'M61332', 'M61339', 'M6134', 'M61341', 'M61342', 'M61349', 'M6135', 'M61351', 'M61352', 'M61359', 'M6136', 'M61361', 'M61362', 'M61369', 'M6137', 'M61371', 'M61372', 'M61379', 'M6138', 'M6139', 'R12', 'T3110', 'T3111', 'T3120', 'T3121', 'T3122', 'T3130', 'T3131', 'T3132', 'T3133', 'T3140', 'T3141', 'T3142', 'T3143', 'T3144', 'T3150', 'T3151', 'T3152', 'T3153', 'T3154', 'T3155', 'T3160', 'T3161', 'T3162', 'T3163', 'T3164', 'T3165', 'T3166', 'T3170', 'T3171', 'T3172', 'T3173', 'T3174', 'T3175', 'T3176', 'T3177', 'T3180', 'T3181', 'T3182', 'T3183', 'T3184', 'T3185', 'T3186', 'T3187', 'T3188', 'T3190', 'T3191', 'T3192', 'T3193', 'T3194', 'T3195', 'T3196', 'T3197', 'T3198', 'T3199', 'V902', 'V9020', 'V9020XA', 'V9020XD', 'V9020XS', 'V9021', 'V9021XA', 'V9021XD', 'V9021XS', 'V9022', 'V9022XA', 'V9022XD', 'V9022XS', 'V9023', 'V9023XA', 'V9023XD', 'V9023XS', 'V9024', 'V9024XA', 'V9024XD', 'V9024XS', 'V9025', 'V9025XA', 'V9025XD', 'V9025XS', 'V9026', 'V9026XA', 'V9026XD', 'V9026XS', 'V9027', 'V9027XA', 'V9027XD', 'V9027XS', 'V9028', 'V9028XA', 'V9028XD', 'V9028XS', 'V9029', 'V9029XA', 'V9029XD', 'V9029XS', 'V931', 'V9310', 'V9310XA', 'V9310XD', 'V9310XS', 'V9311', 'V9311XA', 'V9311XD', 'V9311XS', 'V9312', 'V9312XA', 'V9312XD', 'V9312XS', 'V9313', 'V9313XA', 'V9313XD', 'V9313XS', 'V9314', 'V9314XA', 'V9314XD', 'V9314XS', 'V9319', 'V9319XA', 'V9319XD', 'V9319XS', 'X002', 'X002XXA', 'X002XXD', 'X002XXS', 'X003', 'X003XXA', 'X003XXD', 'X003XXS', 'X004', 'X004XXA', 'X004XXD', 'X004XXS', 'X005', 'X005XXA', 'X005XXD', 'X005XXS', 'X022', 'X022XXA', 'X022XXD', 'X022XXS', 'X023', 'X023XXA', 'X023XXD', 'X023XXS', 'X024', 'X024XXA', 'X024XXD', 'X024XXS', 'X025', 'X025XXA', 'X025XXD', 'X025XXS', 'X0800', 'X0800XA', 'X0800XD', 'X0800XS', 'X0801', 'X0801XA', 'X0801XD', 'X0801XS', 'X0809', 'X0809XA', 'X0809XD', 'X0809XS', 'X0810', 'X0810XA', 'X0810XD', 'X0810XS', 'X0811', 'X0811XA', 'X0811XD', 'X0811XS', 'X0819', 'X0819XA', 'X0819XD', 'X0819XS', 'X0820', 'X0820XA', 'X0820XD', 'X0820XS', 'X0821', 'X0821XA', 'X0821XD', 'X0821XS', 'X0829', 'X0829XA', 'X0829XD', 'X0829XS')


df_cohort_burns.drop_duplicates()
df_cohort_burns.shape[0] #there are 2244 burns patiens

# filter burns patients from cohort_stays_septic_immune
burns = df_cohort_burns['subject_id'].tolist()

df_cohort_stays_septic_immune_burns = df_cohort_stays_septic_immune[df_cohort_stays_septic_immune.subject_id.isin(burns)]

df_cohort_stays_septic_immune_burns.shape[0] # now new patients cohort after filtering burns patients: 415

# figure out where to find fluid information, should be in ICU inputevents and outputevents
 
#check d_items for id
 
%%bigquery df_icu_itemns --project big-cargo-439707-b2
   SELECT *
   FROM physionet-data.mimiciv_icu.d_items
   WHERE abbreviation LIKE '%NaCl%'
   #OR 'label' LIKE '%tacrolimus%'


%%bigquery df_inputevents --project big-cargo-439707-b2
   SELECT *
   FROM physionet-data.mimiciv_icu.inputevents
   WHERE itemid IN (225158, 225159, 225161, 228341)  # input NaCl
   AND subject_id = 13549234 #look at one patient


#check id's that have output events and map that agains cohort_stays_septic
%%bigquery df_outputevents_id --project big-cargo-439707-b2
   SELECT *
   FROM physionet-data.mimiciv_icu.outputevents

df_outputevents_id.drop_duplicates()
df_outputevents_id.shape[0] #4234967 patients have outputevents recorded

# filter output patients from cohort_stays_septic_immune_burns

output_id = df_outputevents_id['subject_id'].tolist()

df_cohort_stays_septic_output = df_cohort_stays_septic_immune_burns[df_cohort_stays_septic_immune_burns.subject_id.isin(output_id)]

df_cohort_stays_septic_output.shape[0] # now new patients cohort after filtering outputevents patients: 413 // from cohort_septic_stays: 31940

# select an example to check input and output events
df_cohort_stays_septic_output = df_cohort_stays_septic_output.sort_values(by=['los_icu'], ascending=False)
df_cohort_stays_septic_output

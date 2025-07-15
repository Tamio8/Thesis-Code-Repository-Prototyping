'''COHORT DEFINITION AND EXTRACTION FOR SIMILARITY PROTOTYPE
This notebook documents the step-by-step filtering process used to define a cohort of
ICU patients from the MIMIC-IV dataset, for use in a similarity-based decision support prototype.

The primary dataset source is the MIMIC-IV database, accessed via Google BigQuery in Google Colab.
The process includes authentication, cohort definition, and export of relevant patient information (demographics, medications, vitals, diagnoses, and discharge notes).

Steps followed:

1. Authenticate and connect to BigQuery using project ID big-cargo-439707-b2.
2. Initial cohort: Adult ICU patients with >= 24h ICU stay and a documented diagnosis of sepsis-3.
3. Exclusion: Patients with immunocompromising conditions (based on ICD codes used in prior literature) were excluded.
4. Subgroup selection: Focused on patients with burn injuries based on a wide range of relevant ICD codes.
5. Medication and vitals: Extracted medication records for curiosity (e.g., corticosteroids, calcineurin inhibitors; not used in prototype),
    vital signs (including glucose), and discharge summaries for these patients.
6. Exported datasets: Saved cleaned subsets for further prototype development.

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

#  This query defines the cohort_stay used for prototype 2, similarity dashboard
#  Inclusion criteria:
#   adult `physionet-data.mimiciv_hosp.patients`
#   stay in ICU `physionet-data.mimiciv_icu`
#   In ICU for at least 24 hours
#   First ICU admission
# 
%%bigquery df_cohort_stays_septic --project big-cargo-439707-b2
 
   SELECT subject_id, hadm_id, stay_id, gender, dod, admittime, dischtime, los_hospital, admission_age, race, icu_intime, icu_outtime, los_icu
     FROM physionet-data.mimiciv_derived.icustay_detail
     WHERE admission_age >= 18.0 AND los_icu > 1.0 AND first_icu_stay = TRUE # age, length of stay, first icu stay
        AND subject_id IN (SELECT subject_id
                            FROM physionet-data.mimiciv_derived.sepsis3
                            WHERE sepsis3.sepsis3 = TRUE) # sepsis 3 true

df_cohort_stays_septic.drop_duplicates()
df_cohort_stays_septic.shape[0] #number of patients in cohort_stay alone without sepsis is 51376


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
   SELECT hadm_id
   FROM physionet-data.mimiciv_hosp.diagnoses_icd
   WHERE icd_code in ('042-0449', '20000-20238', '20250-20301', '2386', '2733','20302-20382', '1960-1991', '20970-20975', '20979', '78951', '1400-1729', '1740-1759',
   '179-1958', '20900-20924', '20925-2093', '20930-20936', '25801-25803', '99680-99689', '23877', '1992', 'V420-V427', 'V4281-V4284', 'V4289', 'V429',
   '5550-5552', '35800-35801', '7100-7105', '7108-7109', '5565-5566', '5568-5569', '7010', '7140', '5559', '340', '7200', '6960', '35971', '4465')

df_cohort_immunecompromised.drop_duplicates()
df_cohort_immunecompromised.shape[0] #there are 9554 immunocompromised patiens

# filter immunocompromised patients from cohort_stays_septic

immunecompromised = df_cohort_immunecompromised['hadm_id'].tolist()

df_cohort_stays_septic_immune = df_cohort_stays_septic[~df_cohort_stays_septic.hadm_id.isin(immunecompromised)]

df_cohort_stays_septic_immune.shape[0] # now new patients cohort after excluding immuncompromised patients: 30575


#A decision was made to focus on patients with burn injuries, informed by the clinical specialisation of one of the collaborating hospitals.
#This focus also served to delineate a more manageable and clinically meaningful cohort for the purposes of this study.
 
#find ICD code for burns
 %%bigquery df_icd_burns --project big-cargo-439707-b2
 
   SELECT * FROM physionet-data.mimiciv_hosp.d_icd_diagnoses
   WHERE long_title LIKE '%burn%'

# there are 334 diagnosis that include 'burn', some might not be relevant like maybe sunburn, depending on degree
burns_icd = df_icd_burns['icd_code'].tolist()

print(burns_icd)


# check how many of those patients have a burns diagnosis
%%bigquery df_cohort_burns --project big-cargo-439707-b2
   SELECT hadm_id
   FROM physionet-data.mimiciv_hosp.diagnoses_icd
   WHERE icd_code in ('69271', '69276', '69277', '7871', '9065', '9066', '9067', '9068', '9069', '9100', '9101', '9110', '9111', '9120', '9121', '9130', '9131', '9140', '9141', '9150', '9151', '9160', '9161', '9170', '9171', '9190', '9191', '9400', '9401', '9402', '9403', '9404', '9409', '94800', '94810', '94811', '94820', '94821', '94822', '94830', '94831', '94832', '94833', '94840', '94841', '94842', '94843', '94844', '94850', '94851', '94852', '94853', '94854', '94855', '94860', '94861', '94862', '94863', '94864', '94865', '94866', '94870', '94871', '94872', '94873', '94874', '94875', '94876', '94877', '94880', '94881', '94882', '94883', '94884', '94885', '94886', '94887', '94888', '94890', '94891', '94892', '94893', '94894', '94895', '94896', '94897', '94898', '94899', 'E8030', 'E8031', 'E8032', 'E8033', 'E8038', 'E8039', 'E8370', 'E8371', 'E8372', 'E8373', 'E8374', 'E8375', 'E8376', 'E8377', 'E8378', 'E8379', 'E8980', 'E8981', 'E9581', 'E9881', 'L55', 'L550', 'L551', 'L552', 'L559', 'M613', 'M6130', 'M6131', 'M61311', 'M61312', 'M61319', 'M6132', 'M61321', 'M61322', 'M61329', 'M6133', 'M61331', 'M61332', 'M61339', 'M6134', 'M61341', 'M61342', 'M61349', 'M6135', 'M61351', 'M61352', 'M61359', 'M6136', 'M61361', 'M61362', 'M61369', 'M6137', 'M61371', 'M61372', 'M61379', 'M6138', 'M6139', 'R12', 'T3110', 'T3111', 'T3120', 'T3121', 'T3122', 'T3130', 'T3131', 'T3132', 'T3133', 'T3140', 'T3141', 'T3142', 'T3143', 'T3144', 'T3150', 'T3151', 'T3152', 'T3153', 'T3154', 'T3155', 'T3160', 'T3161', 'T3162', 'T3163', 'T3164', 'T3165', 'T3166', 'T3170', 'T3171', 'T3172', 'T3173', 'T3174', 'T3175', 'T3176', 'T3177', 'T3180', 'T3181', 'T3182', 'T3183', 'T3184', 'T3185', 'T3186', 'T3187', 'T3188', 'T3190', 'T3191', 'T3192', 'T3193', 'T3194', 'T3195', 'T3196', 'T3197', 'T3198', 'T3199', 'V902', 'V9020', 'V9020XA', 'V9020XD', 'V9020XS', 'V9021', 'V9021XA', 'V9021XD', 'V9021XS', 'V9022', 'V9022XA', 'V9022XD', 'V9022XS', 'V9023', 'V9023XA', 'V9023XD', 'V9023XS', 'V9024', 'V9024XA', 'V9024XD', 'V9024XS', 'V9025', 'V9025XA', 'V9025XD', 'V9025XS', 'V9026', 'V9026XA', 'V9026XD', 'V9026XS', 'V9027', 'V9027XA', 'V9027XD', 'V9027XS', 'V9028', 'V9028XA', 'V9028XD', 'V9028XS', 'V9029', 'V9029XA', 'V9029XD', 'V9029XS', 'V931', 'V9310', 'V9310XA', 'V9310XD', 'V9310XS', 'V9311', 'V9311XA', 'V9311XD', 'V9311XS', 'V9312', 'V9312XA', 'V9312XD', 'V9312XS', 'V9313', 'V9313XA', 'V9313XD', 'V9313XS', 'V9314', 'V9314XA', 'V9314XD', 'V9314XS', 'V9319', 'V9319XA', 'V9319XD', 'V9319XS', 'X002', 'X002XXA', 'X002XXD', 'X002XXS', 'X003', 'X003XXA', 'X003XXD', 'X003XXS', 'X004', 'X004XXA', 'X004XXD', 'X004XXS', 'X005', 'X005XXA', 'X005XXD', 'X005XXS', 'X022', 'X022XXA', 'X022XXD', 'X022XXS', 'X023', 'X023XXA', 'X023XXD', 'X023XXS', 'X024', 'X024XXA', 'X024XXD', 'X024XXS', 'X025', 'X025XXA', 'X025XXD', 'X025XXS', 'X0800', 'X0800XA', 'X0800XD', 'X0800XS', 'X0801', 'X0801XA', 'X0801XD', 'X0801XS', 'X0809', 'X0809XA', 'X0809XD', 'X0809XS', 'X0810', 'X0810XA', 'X0810XD', 'X0810XS', 'X0811', 'X0811XA', 'X0811XD', 'X0811XS', 'X0819', 'X0819XA', 'X0819XD', 'X0819XS', 'X0820', 'X0820XA', 'X0820XD', 'X0820XS', 'X0821', 'X0821XA', 'X0821XD', 'X0821XS', 'X0829', 'X0829XA', 'X0829XD', 'X0829XS')


df_cohort_burns.drop_duplicates()
df_cohort_burns.shape[0] #there are 2244 burns patiens (any type of burn)

# filter burns patients from cohort_stays_septic_immune
burns = df_cohort_burns['hadm_id'].tolist()

df_cohort_stays_septic_immune_burns = df_cohort_stays_septic_immune[df_cohort_stays_septic_immune.hadm_id.isin(burns)]

df_cohort_stays_septic_immune_burns.shape[0] # now new patients cohort after filtering burns patients: 105

"""Use this patient cohort: df_cohort_stays_septic_immune_burns"""

#next, check what medication that patient cohort received
# get table with medication for cohort df_cohort_stays_septic_immune_burns

#safe hadm_id to list
cohort_hadm_id = df_cohort_stays_septic_immune_burns['hadm_id'].tolist()


# get inputevents of that cohort
 
 %%bigquery df_cohort_medication --project big-cargo-439707-b2
 
   SELECT * FROM physionet-data.mimiciv_icu.inputevents
   WHERE hadm_id IN (28041924, 28718343, 29540900, 27897078, 25773534, 28411600, 29776503, 29452084, 23536194, 23494986, 22886008, 26200962, 27971490, 20118852, 26775043, 23827019, 28205137, 26425993, 25218646, 25167492, 22878483, 27314605, 25767268, 24558730, 22485233, 23589164, 28270702, 23089229, 21408401, 23853496, 27016404, 21289627, 21028697, 26642139, 29616446, 25104188, 22794578, 29911181, 24945841, 23480652, 20082646, 26810206, 29765043, 24509535, 21816832, 25680139, 28376556, 22556858, 28444399, 20330892, 20646579, 20055820, 24124338, 26610171, 24138768, 22465051, 20200454, 21868479, 25943593, 21338291, 29877040, 29866696, 28654758, 21397883, 23198300, 21531668, 22270452, 22843593, 26682645, 28678799, 22170965, 20683972, 26709643, 29825359, 24594046, 21694310, 22125536, 23080550, 27122449, 21183102, 24073955, 29557259, 23721563, 28859642, 29130616, 28819981, 20987629, 20611631, 29079804, 23992005, 23939253, 20935702, 26250746, 21090681, 25582099, 22480764, 25769697, 24514478, 29521701, 29638371, 23417361, 23160199, 27138544, 22155829, 23617269)


# replace itemid in df_cohort_medication with name of medication
 %%bigquery df_itemID --project big-cargo-439707-b2
    SELECT itemid, label FROM physionet-data.mimiciv_icu.d_items

df_cohort_medication = df_cohort_medication.merge(df_itemID, on='itemid', how='left')


#Decision to check out how many patients from the cohort received corticosteroids
# icd codes for corticosteroids
 
%%bigquery df_corticosteroids --project big-cargo-439707-b2
    SELECT * FROM physionet-data.mimiciv_hosp.d_hcpcs
    WHERE long_description LIKE '%corticosteroids%'


# select medicated patients: corticosteroids
%%bigquery df_steroids --project big-cargo-439707-b2
   SELECT *
   FROM physionet-data.mimiciv_hosp.pharmacy #table for prescribed medication
   WHERE medication LIKE '%cortisone%' # cortisone, hydrocortisone and prednisone
   OR medication LIKE '%hydrocortisone%'
   OR medication LIKE '%prednisone%'
 
 
   #('G9467', 'G8859', 'G8860', 'G9469') icd codes


# filter corticosteroids patients from cohort_stays_septic_immune_burns

corticosteroids = df_steroids['hadm_id'].tolist()

df_cohort_stays_septic_immune_burns_steroids = df_cohort_stays_septic_immune_burns[df_cohort_stays_septic_immune_burns.hadm_id.isin(corticosteroids)]

df_cohort_stays_septic_immune_burns_steroids.shape[0] # now new patients cohort after filtering corticosteroids patients: 6


# instead of corticosteroids, check how many received calcineurin inhibitors: cyclosporine, tacrolimus
 
# select medicated patients: calcineurin inhibitors
%%bigquery df_calci --project big-cargo-439707-b2
   SELECT hadm_id
   FROM physionet-data.mimiciv_hosp.pharmacy #table for prescribed medication
   WHERE medication LIKE '%cyclosporine%'
   OR medication LIKE '%tacrolimus%'

df_calci.drop_duplicates()
df_calci.shape[0] # 114

# filter calcineurin inhibitors patients from cohort_stays_septic_immune_burns

calcineurin = df_calci['hadm_id'].tolist()

df_cohort_stays_septic_immune_burns_calci = df_cohort_stays_septic_immune_burns[df_cohort_stays_septic_immune_burns.hadm_id.isin(calcineurin)]

df_cohort_stays_septic_immune_burns_calci.shape[0] # now new patients cohort after filtering calcineurin inhibitors patients: 3

# check how many from cohort_stays_septic received calcineurin inhibitors

#df_cohort_stays_septic_immune_calci = df_cohort_stays_septic_immune[df_cohort_stays_septic_immune.subject_id.isin(calcineurin)]

#df_cohort_stays_septic_immune_calci.shape[0] # now new patients cohort after filtering calcineurin inhibitors patients: 37

#For vital signs: mimiciv_derived.vitalsign of patients in df_cohort_stays_septic_immune_burns

df_cohort_stays_septic_immune_burns.describe() # check missing values

df_cohort_stays_septic_immune_burns.info()

"""looks very complete"""

cohort_ID = df_cohort_stays_septic_immune_burns['stay_id'].tolist()
#print(cohort_ID)


# select vitalsigns for df_cohort_stays_septic_immune_burns
 %%bigquery df_cohort_vitalsign --project big-cargo-439707-b2
 SELECT * FROM physionet-data.mimiciv_derived.vitalsign WHERE stay_id IN (37171670, 30006565, 39768103, 34253386, 31885114, 38486717, 35255147, 36356955, 36974556, 30624987, 38658006, 38372401, 30637692, 33271298, 31676535, 32416606, 33187188, 38330612, 39449136, 36279134, 38508770, 31578659, 31185496, 36837185, 37047409, 30602234, 32620190, 32313856, 34132590, 34634419, 37744846, 31041981, 39150944, 30626122, 35776095, 37612264, 32031913, 33832160, 30696453, 39711363, 31986636, 37787511, 30774936, 34564898, 37000271, 34818547, 38482726, 39743745, 32389325, 39672431, 31557612, 30162054, 35709738, 39754636, 33035558, 33439987, 37609236, 33485562, 34473161, 38874499, 38192611, 36299350, 30698001, 39198142, 30782843, 33919649, 31164656, 38671812, 31983448, 35881882, 36148917, 33882509, 31976189, 31847457, 30045760, 36805048, 31985637, 35146959, 38315255, 38227266, 31047765, 36557642, 36663037, 37259563, 37766634, 32404823, 30012063, 32941531, 35580228, 33757419, 36901358, 36780485, 31588174, 37189090, 34607754, 32452859, 36506153, 38356273, 32319147, 33907802, 32368927, 34796801, 37678620, 38536869, 36552784)

df_cohort_vitalsign.info()

# see how many IDs have non-null in glucose

df_cohort_vitalsign_glucose = df_cohort_vitalsign.loc[df_cohort_vitalsign['glucose'].notnull(), ['subject_id','stay_id', 'charttime','heart_rate', 'sbp', 'dbp', 'mbp', 'sbp_ni', 'dbp_ni', 'mbp_ni', 'resp_rate', 'temperature', 'temperature_site', 'spo2', 'glucose'  ]]

df_cohort_vitalsign_glucose.info() # note that values measured in different frequency

df_cohort_vitalsign_glucose['subject_id'].nunique()

df_cohort_vitalsign_glucose.shape[0] - df_cohort_vitalsign_glucose.dropna().shape[0]


#THOUGHTS How to deal with missing values? Maybe not measured as often? If value not there, then automatically not similar to a patient? Cannot just delete because missing value also has meaning?

# get discharge summaries from cohort ids
%%bigquery df_cohort_discharge_summary --project big-cargo-439707-b2
 SELECT * FROM physionet-data.mimiciv_note.discharge WHERE subject_id IN (10148533, 15479218, 13804231, 15383083, 18614648, 14635008, 18877846, 19908221, 11233081, 15694968, 16044793, 13267235, 17589058, 14120635, 18888515, 18172425, 17524724, 19832679, 15248985, 14603776, 16076363, 19509653, 17170716, 15526304, 15752366, 15716202, 15579080, 17760909, 19970265, 13179092, 14645595, 12627644, 11166681, 19906564, 14581489, 12798063, 17760270, 11480283, 14098557, 14482934, 13869899, 15130525, 16103537, 19387278, 12710598, 11258077, 12798506, 18212558, 14198265, 12043836, 19482563, 10992347, 16952444, 19348626, 15299779, 14435499, 17209535, 13146468, 14275115, 17310431, 17447711, 12802332, 13549234, 11294917, 11900074, 14834230, 13050394, 13873595, 11376915, 10598816, 10935252, 11959467, 19311178, 14717765, 11536174, 13558665, 16285428, 13505628, 15703508, 14635022, 10644961, 18870809, 16517380, 16099802, 10933609, 17233716, 12621159, 10230202, 17792869, 17823750, 11425766, 18227829, 14976423, 16529186, 14280430, 13892963, 18629932, 15524760, 17546877, 14250712, 15712741, 14040144, 19043108, 10599715)

# get discharge summaries from cohort ids
 
%%bigquery df_cohort_discharge_summary_detail --project big-cargo-439707-b2
 SELECT * FROM physionet-data.mimiciv_note.discharge_detail WHERE subject_id IN (10148533, 15479218, 13804231, 15383083, 18614648, 14635008, 18877846, 19908221, 11233081, 15694968, 16044793, 13267235, 17589058, 14120635, 18888515, 18172425, 17524724, 19832679, 15248985, 14603776, 16076363, 19509653, 17170716, 15526304, 15752366, 15716202, 15579080, 17760909, 19970265, 13179092, 14645595, 12627644, 11166681, 19906564, 14581489, 12798063, 17760270, 11480283, 14098557, 14482934, 13869899, 15130525, 16103537, 19387278, 12710598, 11258077, 12798506, 18212558, 14198265, 12043836, 19482563, 10992347, 16952444, 19348626, 15299779, 14435499, 17209535, 13146468, 14275115, 17310431, 17447711, 12802332, 13549234, 11294917, 11900074, 14834230, 13050394, 13873595, 11376915, 10598816, 10935252, 11959467, 19311178, 14717765, 11536174, 13558665, 16285428, 13505628, 15703508, 14635022, 10644961, 18870809, 16517380, 16099802, 10933609, 17233716, 12621159, 10230202, 17792869, 17823750, 11425766, 18227829, 14976423, 16529186, 14280430, 13892963, 18629932, 15524760, 17546877, 14250712, 15712741, 14040144, 19043108, 10599715)


stay_id = df_cohort_vitalsign.stay_id.unique().tolist()
print(stay_id)

#look at diagnosis for cohort
%%bigquery df_cohort_diagnosis --project big-cargo-439707-b2
   SELECT diagnoses_icd.subject_id, diagnoses_icd.hadm_id, diagnoses_icd.icd_code, d_icd_diagnoses.long_title
     FROM physionet-data.mimiciv_hosp.diagnoses_icd, physionet-data.mimiciv_hosp.d_icd_diagnoses
     WHERE subject_id IN (10148533, 15479218, 13804231, 15383083, 18614648, 14635008, 18877846, 19908221, 11233081, 15694968, 16044793, 13267235, 17589058, 14120635, 18888515, 18172425, 17524724, 19832679, 15248985, 14603776, 16076363, 19509653, 17170716, 15526304, 15752366, 15716202, 15579080, 17760909, 19970265, 13179092, 14645595, 12627644, 11166681, 19906564, 14581489, 12798063, 17760270, 11480283, 14098557, 14482934, 13869899, 15130525, 16103537, 19387278, 12710598, 11258077, 12798506, 18212558, 14198265, 12043836, 19482563, 10992347, 16952444, 19348626, 15299779, 14435499, 17209535, 13146468, 14275115, 17310431, 17447711, 12802332, 13549234, 11294917, 11900074, 14834230, 13050394, 13873595, 11376915, 10598816, 10935252, 11959467, 19311178, 14717765, 11536174, 13558665, 16285428, 13505628, 15703508, 14635022, 10644961, 18870809, 16517380, 16099802, 10933609, 17233716, 12621159, 10230202, 17792869, 17823750, 11425766, 18227829, 14976423, 16529186, 14280430, 13892963, 18629932, 15524760, 17546877, 14250712, 15712741, 14040144, 19043108, 10599715)
       AND  diagnoses_icd.icd_code = d_icd_diagnoses.icd_code

# safe demographics as csv
df_cohort_stays_septic_immune_burns.to_csv('df_cohort_stays_septic_immune_burns')

#safe steroids as csv
df_cohort_stays_septic_immune_burns_steroids.to_csv('df_cohort_stays_septic_immune_burns_steroids') #not used in the end

# safe vital signs as csv
df_cohort_vitalsign.to_csv('df_cohort_vitalsign')

# safe vital signs glucose as csv
df_cohort_vitalsign_glucose.to_csv('df_cohort_vitalsign_glucose')

# safe steroids mediaction as csv
df_steroids.to_csv('df_steroids')

# safe medication as csv
df_cohort_medication.to_csv('df_medication')

# safe discharge notes
df_cohort_discharge_summary.to_csv('df_discharge_note')
df_cohort_discharge_summary_detail.to_csv('df_discharge_note_detail')

# safe diagnosis of cohort
df_cohort_diagnosis.to_csv('df_diagnosis')

"""# IMPORT CSV AND RUN NOTEBOOK FROM HERE#"""

import pandas as pd
# read safed csv
df_cohort_stays_septic_immune_burns = pd.read_csv('/content/df_cohort_stays_septic_immune_burns')
df_cohort_stays_septic_immune_burns_steroids = pd.read_csv('/content/df_cohort_stays_septic_immune_burns_steroids')
df_cohort_vitalsign = pd.read_csv('/content/df_cohort_vitalsign')
df_cohort_vitalsign_glucose = pd.read_csv('/content/df_cohort_vitalsign_glucose')
df_steroids = pd.read_csv('/content/df_steroids')

"""There are 415 patients in the cohort.
1.   pick one ID: 14645595(no steroids)
2.   get vital signs
3. plot vital signs
4. get medication
5. plot medication
"""

df_cohort_stays_septic_immune_burns.head()

df_cohort_vitalsign.head()

#get all vitalsigns for ID #14645595

nums = [14645595]

df_14645595_vitalsign = df_cohort_vitalsign[df_cohort_vitalsign['subject_id'].isin(nums)]
df_14645595_vitalsign['charttime'] = pd.to_datetime(df_14645595_vitalsign['charttime'])
df_14645595_vitalsign = df_14645595_vitalsign.sort_values(by='charttime')
df_14645595_vitalsign.head()

df_14645595_vitalsign_heartrate = df_14645595_vitalsign[['charttime', 'heart_rate']]
df_14645595_vitalsign_heartrate.info()

# plot heart rate of ID 14645595
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.figure(figsize=(12, 6))
plt.plot(df_14645595_vitalsign_heartrate['charttime'], df_14645595_vitalsign_heartrate['heart_rate'], marker='o', linestyle='-', color='b')
# Formatting the x-axis to show both date and time
date_format = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
plt.gca().xaxis.set_major_formatter(date_format)
# Adjusting the x-axis ticks to show more frequent intervals
plt.gca().xaxis.set_major_locator(mdates.SecondLocator(interval=14400))#every 4h in seconds
#Set Labels
plt.title('Plot of Heartrate Over Time for #14645595')
plt.xlabel('Datetime')
plt.ylabel('HR')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()

"""SELECT CORTICOSTEROIDS PATIENT
14645595
"""

df_cohort_stays_septic_immune_burns_steroids.head(20)

#check the distribution of HR across the cohort, generate a dataframe of: ID, day, time, measurement
#for day: number the days starting from day 1
#patient cohort: df_cohort_stays_septic_immune_burns, 415 patients

df_cohort_stays_septic_immune_burns.head()

df_cohort_vitalsign.head()

#create table for heartrate only for all patients
df_cohort_vitalsign_HR = df_cohort_vitalsign[df_cohort_vitalsign['heart_rate'].notna()][['stay_id', 'charttime', 'heart_rate']]
df_cohort_vitalsign_HR.head()

df_cohort_vitalsign_HR.nunique()

# Group by ID and normalize datetime
df_cohort_vitalsign_HR_normalised = []
for id, group in df_cohort_vitalsign_HR.groupby('stay_id'):
    sorted_group = group.sort_values('charttime')
    normalized = sorted_group.reset_index()
    normalized['stay_id'] = id
    normalized['charttime'] = normalized.index + 1  # Start from day 1
    df_cohort_vitalsign_HR_normalised.append(normalized)

df_cohort_vitalsign_HR_normalised = pd.concat(df_cohort_vitalsign_HR_normalised)

#Reshape Data for Plotting

df_cohort_vitalsign_HR_normalised_melted = df_cohort_vitalsign_HR_normalised.melt(id_vars=['stay_id', 'charttime'], var_name='heart_rate', value_name='datetime_value')

# normalise datetime per ID
df_cohort_vitalsign_HR['charttime'] = pd.to_datetime(df_cohort_vitalsign_HR['charttime'])
df_cohort_vitalsign_HR['relative_time'] = df_cohort_vitalsign_HR.groupby('stay_id')['charttime'].transform(lambda x: (x - x.min()).dt.total_seconds() / 3600)  # Convert to hours

# Plot
plt.figure(figsize=(12, 6))
for key, grp in df_cohort_vitalsign_HR.groupby('stay_id'):
    plt.scatter(grp['relative_time'], grp['heart_rate'], label=f'stay_ID {key}', alpha=0.7)

# Formatting
plt.xlabel("Time since first measurement (hours)")
plt.ylabel("Heartrate")
plt.title("Distribution of Heartrate Across Normalized Time")
plt.legend(title="stay_id", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()

#Measurement Frequency of Vital Signs in Dataset:


#   heartrate: every 1h
#   sbp/dbp/mbp: irregular, but approx every 1h
# resp_rate: every 1h
# temperature every 3-6h
# spo2 more frequently, maybe every 30min
# glucose irregular, maybe every 3h



# Heart Rate - Min: 28.0, Max: 179.0, Mean: 87.74882855392562
# SBP - Min: 38.0, Max: 231.0, Mean: 121.60306218621992
# DPB - Min: 19.0, Max: 202.0, Mean: 65.258420785866
# Respiratory Rate - Min: 2.0, Max: 53.0, Mean: 18.887376845763388
# Temperature - Min: 33.6, Max: 40.5, Mean: 37.050380874349976
# SPO2 - Min: 41.0, Max: 100.0, Mean: 96.74076789628943
# Glucose - Min: 42.0, Max: 467.0, Mean: 131.7117074089331

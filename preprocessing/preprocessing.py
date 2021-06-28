import pandas as pd
import numpy as np

from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split

def _load_files(patients_path: str, tumours_path: str, treatments_path: str, thgebeurtenis_path: str) -> tuple:
    """
    Loads patient, tumour and treatments data
    """
    # Load data
    patients = pd.read_csv(patients_path)
    tumours = pd.read_csv(tumours_path)
    treatments = pd.read_csv(treatments_path)
    th_gebeurtenis = pd.read_csv(thgebeurtenis_path, engine='python')

    return patients, tumours, treatments, th_gebeurtenis

def _filter_treatments_tumours(tumours: pd.DataFrame, treatments: pd.DataFrame) -> pd.DataFrame:
    """
    Filter reatments on tumours not in tumours table,
    treatments that occur more than 12 months after tumour incidence, 
    tumours that do not have any registered treatments, and 
    tumours that are not first episode per patient
    """
    # Set datetime type
    treatments['gbs_begin_dtm'] = pd.to_datetime(treatments['gbs_begin_dtm'])
    treatments['gbs_eind_dtm'] = pd.to_datetime(treatments['gbs_eind_dtm'])

    # Remove treatments not in tumours table
    to_keep = pd.DataFrame(tumours['eid'])
    treatments = treatments.merge(to_keep, on='eid', how='inner')

    # Remove tumours without treatments
    tumours = tumours[tumours['eid'].isin(treatments['eid'].unique())]

    # Remove tumours that are not the first episode per patient
    tumours = tumours[tumours['eerste_episode']==1]

    # Keep only one tumour (first) per patient
    tumours = tumours.sort_values('tum_incidentie_dtm')
    tumours = tumours.drop_duplicates('rn', keep='first')

    # Remove treatments with start date longer than 12 months after incidence
    tumours['tum_incidentie_dtm'] = pd.to_datetime(tumours['tum_incidentie_dtm'])

    # Calculate months after incidence
    treatments_incidence = treatments.merge(tumours[['eid', 'tum_incidentie_dtm']])
    treatments_incidence['time_after_incidence'] = treatments_incidence['gbs_begin_dtm'] - treatments_incidence['tum_incidentie_dtm']
    treatments_incidence['time_after_incidence'] = treatments_incidence['time_after_incidence'].dt.days
    treatments_incidence['months_after_incidence'] = treatments_incidence['time_after_incidence'] / 30.5

    # Only keep treatments within 12 months after incidence
    treatments_within_12 = treatments_incidence[(treatments_incidence['months_after_incidence']<=12) | (treatments_incidence['months_after_incidence'].isnull())]
    treatments = treatments[treatments['gid'].isin(treatments_within_12['gid'])]
    
    # Remove if this leads to gaps in gbs_vnr (due to missing treatment dates of treatments occurring in order 
    # after treatments known to occur after 12 months (with date))
    gids_to_remove = treatments[treatments['gbs_vnr']!= treatments.groupby('eid').cumcount()+1]['gid']
    treatments = treatments[~treatments['gid'].isin(gids_to_remove)]

    return tumours, treatments

def _get_treatments_per_tumour(treatments_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Gets as input either filtered treatments or sequential treatments and generates 1 row per tumour
    Includes columns nr_treatments and treatments
    """
    treatments_per_tumour = treatments_dataframe.groupby('eid').agg({'rn': 'count',
                                                                     'gbs_gebeurtenis_code': lambda x: list(x)}).rename(
                                                           columns={'rn': 'nr_treatments',
                                                                     'gbs_gebeurtenis_code': 'treatments'})

    return treatments_per_tumour      

def _get_relevant_features(treatments: pd.DataFrame, tumours: pd.DataFrame, patients: pd.DataFrame) -> tuple:
    """
    Keeps only features needed for generation for any of the algorithms
    """
    treatments = treatments[['eid', 'gid', 'gbs_gebeurtenis_code', 'gbs_vnr']].copy()

    tumours = tumours[['rn', 'eid', 'tum_incidentie_dtm', 'tum_topo_code', 'tum_differentiatiegraad_code', 
                               'tum_lymfklieren_positief_atl', 'tum_topo_sublokalisatie_code', 'stadium']].copy()

    patients = patients[['rn', 'pat_geboorte_dtm', 'pat_geslacht_code', 'pat_overlijden_dtm']].copy()

    return treatments, tumours, patients

def _keep_till_2017(treatments: pd.DataFrame, tumours: pd.DataFrame, patients: pd.DataFrame) -> tuple:
    """
    Keeps only tumours and associated patients & treatments with incidence date at the latest 31-21-2017
    """
    # Set correct dtype
    tumours['tum_incidentie_dtm'] = pd.to_datetime(tumours['tum_incidentie_dtm'])

    # Filter
    tumours_2017 = tumours[tumours['tum_incidentie_dtm']<='2017-12-31']
    patients_2017 = patients[patients['rn'].isin(np.unique(tumours_2017['rn']))]
    treatments_2017 = treatments[treatments['eid'].isin(np.unique(tumours_2017['eid']))]

    return treatments_2017, tumours_2017, patients_2017

def _merge_tumours_patients(tumours_2017: pd.DataFrame, patients_2017: pd.DataFrame) -> pd.DataFrame:
    """
    Merge tumours and patients table into one table
    """
    tumours_patients_2017 = tumours_2017.merge(patients_2017, on='rn')

    return tumours_patients_2017


def _engineer_static_variables(tumours_patients_2017: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer all static variables to be generated (survival_1 and age_at_diagnosis)
    Also discretize and clean all features
    """
    # SURVIVAL
    # Set dtypes
    tumours_patients_2017['tum_incidentie_dtm'] = pd.to_datetime(tumours_patients_2017['tum_incidentie_dtm'])
    tumours_patients_2017['pat_overlijden_dtm'] = pd.to_datetime(tumours_patients_2017['pat_overlijden_dtm'])

    # Days of survival 
    tumours_patients_2017['days_survival'] = tumours_patients_2017['pat_overlijden_dtm'] - tumours_patients_2017['tum_incidentie_dtm']
    tumours_patients_2017['days_survival'] = tumours_patients_2017['days_survival'].dt.days

    # Binarize survival
    def get_1year_survival(days_survival):
        if days_survival <= 365:
            return 0
        else:
            return 1

    tumours_patients_2017['survival_1'] = tumours_patients_2017['days_survival'].apply(get_1year_survival).astype('str')

    # AGE AT DIAGNOSIS
    # Set data types
    tumours_patients_2017['tum_incidentie_dtm'] = pd.to_datetime(tumours_patients_2017['tum_incidentie_dtm'])
    tumours_patients_2017['pat_geboorte_dtm'] = pd.to_datetime(tumours_patients_2017['pat_geboorte_dtm'])

    # Calculate age at diagnosis in years
    def get_age_in_years(patient):
        age_in_days = patient['tum_incidentie_dtm'] - patient['pat_geboorte_dtm']
        age_in_days = age_in_days.days
        age_in_years = age_in_days/365.25
        return age_in_years

    tumours_patients_2017['age_at_diagnosis'] = tumours_patients_2017.apply(lambda row: get_age_in_years(row), axis=1)
    
    # Discretize
    bins_age = [0, 15, 30, 45, 60, 75, 1000]
    labels_age = ['0-14', '15-29', '30-44', '45-59', '60-75', '75-100']
    tumours_patients_2017['age_at_diagnosis'] = pd.cut(tumours_patients_2017['age_at_diagnosis'], bins=bins_age, labels=labels_age)
    tumours_patients_2017['age_at_diagnosis'] = tumours_patients_2017['age_at_diagnosis'].astype(str)

    # STAGE
    # Replace null stadium by M (unknown)
    tumours_patients_2017['stadium'] = tumours_patients_2017['stadium'].fillna('M')

    # Replace NVT by M (unknown)
    tumours_patients_2017['stadium'] = tumours_patients_2017['stadium'].replace('NVT', 'M')

    # Function to get final broad stadium classes
    def get_broad_stadium(stadium):
        return stadium[0]

    tumours_patients_2017['stadium'] = tumours_patients_2017['stadium'].apply(get_broad_stadium)

    # DIFFERENTIATION GRADE
    # Replace 5 and 6 to 9 (unknown)
    tumours_patients_2017['tum_differentiatiegraad_code'] = tumours_patients_2017['tum_differentiatiegraad_code'].replace(6, 9)
    tumours_patients_2017['tum_differentiatiegraad_code'] = tumours_patients_2017['tum_differentiatiegraad_code'].replace(5, 9)

    # Change dtype
    tumours_patients_2017['tum_differentiatiegraad_code'] = tumours_patients_2017['tum_differentiatiegraad_code'].astype('int')
    tumours_patients_2017['tum_differentiatiegraad_code'] = tumours_patients_2017['tum_differentiatiegraad_code'].astype('object')

    # SUBLOCATION CODE
    # Change dtype
    tumours_patients_2017['tum_topo_sublokalisatie_code'] = tumours_patients_2017['tum_topo_sublokalisatie_code'].astype('int')
    tumours_patients_2017['tum_topo_sublokalisatie_code'] = tumours_patients_2017['tum_topo_sublokalisatie_code'].astype('object')

    # GENDER
    def change_gender_categorical(gender):
        if gender==1:
            return 'Male'
        else:
            return 'Female'

    tumours_patients_2017['pat_geslacht_code'] = tumours_patients_2017['pat_geslacht_code'].apply(change_gender_categorical)

    # NR POSITIVE LYMPH NODES
    # Discretize
    bins_lymfklieren = [-1, 0.5, 89, 1000]
    labels_lymfklieren = ['0','1-89','unknown']

    tumours_patients_2017['tum_lymfklieren_positief_atl'] = pd.cut(tumours_patients_2017['tum_lymfklieren_positief_atl'], bins_lymfklieren, labels=labels_lymfklieren)
    tumours_patients_2017['tum_lymfklieren_positief_atl'] = tumours_patients_2017['tum_lymfklieren_positief_atl'].fillna('unknown')
    tumours_patients_2017['tum_lymfklieren_positief_atl'] = tumours_patients_2017['tum_lymfklieren_positief_atl'].astype(str)

    # Select only relevant features
    features = ['eid', 'tum_topo_code', 'pat_geslacht_code', 'tum_differentiatiegraad_code', 'tum_lymfklieren_positief_atl',
                'age_at_diagnosis', 'tum_topo_sublokalisatie_code', 'stadium', 'survival_1']

    tumours_patients_2017 = tumours_patients_2017[features].copy()

    return tumours_patients_2017

def _remove_max_treatment_length(treatments_2017):
    """
    Removes all tumours with number of treatments more than 0.99 quantile of number of treatments
    """
    lengths = treatments_2017.groupby('eid')['gbs_gebeurtenis_code'].count()
    max_treatment_length = lengths.quantile(0.99)

    print('maximum treatment length:', max_treatment_length)

    treatments_2017 = treatments_2017[treatments_2017['eid'].isin(
        lengths[lengths <= max_treatment_length].index)]
    
    return treatments_2017

def _remove_infrequent_treatments(treatments_2017: pd.DataFrame, th_gebeurtenis: pd.DataFrame) -> pd.DataFrame:
    """
    Removes tumours with treatments that occur less than 1/1000 times in the data set
    """
    print('infrequent')
    infrequent = treatments_2017['gbs_gebeurtenis_code'].value_counts()[treatments_2017['gbs_gebeurtenis_code'].value_counts(
        normalize=True)<=0.001]

    infrequent_tumours = treatments_2017[treatments_2017['gbs_gebeurtenis_code'].isin(infrequent.index)]['eid']

    treatments_2017 = treatments_2017[~treatments_2017['eid'].isin(infrequent_tumours)]

    return treatments_2017

def _remove_tumours_treatment_filtering(tumours_patients_2017: pd.DataFrame, treatments_2017: pd.DataFrame) -> tuple:
    """
    Removes tumours removed through treatment length and occurrence filtering
    """
    tumours_patients_2017 = tumours_patients_2017[
                                tumours_patients_2017['eid'].isin(treatments_2017['eid'])]

    return tumours_patients_2017

def _get_train_test(tumours_patients_2017: pd.DataFrame, treatments_2017: pd.DataFrame) -> tuple:
    """
    Splits tables into 80-20 train test split stratified on survival
    """
    # Divide in X, y in order to stratify on survival
    X = tumours_patients_2017.drop('survival_1', axis=1)
    y = tumours_patients_2017['survival_1']

    # Stratified train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2021)

    # Keep train and test eid's to make actual split
    train_eid = X_train['eid']
    test_eid = X_test['eid']

    # Tumours patients
    tumours_patients_2017_train = tumours_patients_2017[tumours_patients_2017['eid'].isin(train_eid)].copy()
    tumours_patients_2017_test = tumours_patients_2017[tumours_patients_2017['eid'].isin(test_eid)].copy()

    # Sequential treatments
    treatments_2017_train = treatments_2017[treatments_2017['eid'].isin(train_eid)].copy()
    treatments_2017_test = treatments_2017[treatments_2017['eid'].isin(test_eid)].copy()

    return tumours_patients_2017_train, tumours_patients_2017_test, treatments_2017_train, treatments_2017_test

def _get_treatment_nr(treatments, nr):
    if len(treatments)>=nr:
        return treatments[nr-1]
    else:
        return 'None'
    
def _get_sequential_treatments_df(treatments_grouped, max_nr):
    seq_treatments_df = pd.DataFrame(index=treatments_grouped.index)
    
    for nr in range(1, max_nr+1):
        seq_treatments_df['treatment_' + str(nr)] = treatments_grouped['gbs_gebeurtenis_code'].apply(_get_treatment_nr, 
                                                                                                     args=[nr])
    
    return seq_treatments_df

def _remove_unique_treatments_train(treatments_2017_train, tumours_patients_2017_train, max_nr_treatments):
    """
    Removes unique treatments per treatment column (1-5) in flattened data frame (sequential data pivoting)
    Needed for PrivBayes not to disclose private information immediately 
    """
    # List of treatments sorted by order of occurrence per patient
    treatments_2017_train = treatments_2017_train.sort_values(['eid', 'gbs_vnr'])
    treatments_grouped = pd.DataFrame(treatments_2017_train.groupby('eid')['gbs_gebeurtenis_code'].apply(list))

    # Get flattened version of treatments (5 treatment columns)
    sequential_treatments = _get_sequential_treatments_df(treatments_grouped, max_nr_treatments)

    # Remove unique treatments per column
    for i in range(1, max_nr_treatments+1):
        # Find unique treatments (only occurring once in column)
        vc = sequential_treatments['treatment_'+str(i)].value_counts()
        unique_treatments = vc[vc==1]

        # Find patients with unique treatments and drop from sequential treatments data frame
        patients_unique_treatments = sequential_treatments[sequential_treatments[
            'treatment_'+str(i)].isin(unique_treatments.index)].index
        sequential_treatments = sequential_treatments.drop(patients_unique_treatments)

    # Drop patients with unique treatments from treatments and tumours data frame
    treatments_2017_train = treatments_2017_train[treatments_2017_train['eid'].isin(sequential_treatments.index)]
    tumours_patients_2017_train = tumours_patients_2017_train[tumours_patients_2017_train['eid'].isin(sequential_treatments.index)]

    return treatments_2017_train, tumours_patients_2017_train

def main(patients_path: str, tumours_path: str, treatments_path: str, thgebeurtenis_path: str) -> tuple:
    """
    Preprocesses data
    """
    # Load files
    patients, tumours, treatments, th_gebeurtenis = _load_files(patients_path, tumours_path, treatments_path, thgebeurtenis_path)

    # Filter treatments
    tumours, treatments = _filter_treatments_tumours(tumours, treatments)

    # Get relevant features
    treatments, tumours, patients = _get_relevant_features(treatments, tumours, patients)

    # Keep all data with tumour incidence date at latest 31-12-2017
    treatments_2017, tumours_2017, patients_2017 = _keep_till_2017(treatments, tumours, patients)
    print('Untill 2017')
    print(treatments_2017.shape, tumours_2017.shape)

    # Merge tumour and patient variables into one dataframe
    tumours_patients_2017 = _merge_tumours_patients(tumours_2017, patients_2017)

    # Engineer all static (patient and tumour) variables
    tumours_patients_2017 = _engineer_static_variables(tumours_patients_2017)

    # Max number of treatments per tumour
    treatments_2017 = _remove_max_treatment_length(treatments_2017)
    print(treatments_2017.shape)

    # Change and remove infrequent treatments
    treatments_2017 = _remove_infrequent_treatments(treatments_2017, th_gebeurtenis)
    print('filtered treatment codes')
    print(treatments_2017['gbs_gebeurtenis_code'].value_counts())
    print('Number of unique treatment codes in data set:', len(treatments_2017['gbs_gebeurtenis_code'].value_counts()))

    # Also remove these tumours from static dataframe
    tumours_patients_2017 = _remove_tumours_treatment_filtering(tumours_patients_2017, treatments_2017)

    print('left')
    print(treatments_2017.shape)
    print(tumours_patients_2017.shape)

    # Train test split
    tumours_patients_2017_train, tumours_patients_2017_test, treatments_2017_train, treatments_2017_test = _get_train_test(tumours_patients_2017, treatments_2017)

    # Drop unique treatments per treatment column in training data frames
    treatments_2017_train, tumours_patients_2017_train = _remove_unique_treatments_train(treatments_2017_train, tumours_patients_2017_train, 5)

    print('train test')
    print(tumours_patients_2017_train.shape, tumours_patients_2017_test.shape, treatments_2017_train.shape, treatments_2017_test.shape)
    return tumours_patients_2017_train, tumours_patients_2017_test, treatments_2017_train, treatments_2017_test

if __name__=='__main__':
    TUMOURS_PATH = '../../../Master Thesis/data/crc_tumor.csv'
    TREATMENTS_PATH = '../../../Master Thesis/data/crc_behandeling.csv'
    PATIENTS_PATH = '../../../Master Thesis/data/crc_patient.csv'
    THGEBEURTENIS_PATH = '../../../Master Thesis/data/thesauri/th_gebeurtenis.csv'

    tumours_patients_2017_train, tumours_patients_2017_test, treatments_2017_train, treatments_2017_test = main(PATIENTS_PATH, TUMOURS_PATH, TREATMENTS_PATH, THGEBEURTENIS_PATH)

    tumours_patients_2017_train.to_pickle('../../../Master Thesis/data/preprocessed/tumours_patients_2017_train.pickle')
    tumours_patients_2017_test.to_pickle('../../../Master Thesis/data/preprocessed/tumours_patients_2017_test.pickle')
    treatments_2017_train.to_pickle('../../../Master Thesis/data/preprocessed/treatments_2017_train.pickle')
    treatments_2017_test.to_pickle('../../../Master Thesis/data/preprocessed/treatments_2017_test.pickle')
    
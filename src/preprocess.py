import pandas as pd 
from tqdm import tqdm

from src.helpers import *

def preprocess_admissionDx(**kwargs) -> pd.DataFrame:
    """
    Process the admissionDx data.
    """
    admissionDx_base = get_df('admissionDx')

    split_dfs = admissionDx_base['admitdxpath'].str.split('|', expand=True).iloc[:, 1:]
    split_dfs = {name: group for name, group in split_dfs.groupby(1)}
    for df in split_dfs.values():
        df[SID] = admissionDx_base[SID]

    admissionDx = (split_dfs['All Diagnosis'][[SID, 4, 5]]
                 .rename(columns={4: 'admissionDx_cat', 5: 'admissionDx_str'}))
    admissionDx['operative'] = split_dfs['All Diagnosis'][2] == 'Operative'

    or_within_4hrs = (split_dfs['Was the patient admitted from the O.R. or went to the O.R. within 4 hours of admission?']
                      .set_index(SID)[2].map(lambda x: x=='Yes'))
    admissionDx['or_within_4hrs'] = admissionDx[SID].map(or_within_4hrs)

    non_op_organs = (split_dfs['Non-operative Organ Systems'].set_index(SID)[3])
    admissionDx['non_operative_organs'] = admissionDx[SID].map(non_op_organs)

    op_organs = (split_dfs['Operative Organ Systems'].set_index(SID)[3])
    admissionDx['operative_organs'] = admissionDx[SID].map(op_organs)

    elective = (split_dfs['Elective'].set_index(SID)[2])
    admissionDx['elective'] = admissionDx[SID].map(elective)

    return admissionDx.reset_index(drop=True)

def preprocess_patient(**kwargs) -> pd.DataFrame:
    patient = get_df('patient')
    patient = patient[patient['hospitaldischargeoffset']>=1440]
    patient['mortality'] = patient['unitdischargestatus'] == 'Expired'
    
    patient = patient[[SID, PID, 'hospitalid','gender', 'age',
       'ethnicity', 'admissionheight', 'admissionweight', 'unitstaytype', 'mortality']]
    return patient.reset_index(drop=True)

def preprocess_diagnoses(additional_diagnoses=[], **kwargs) -> pd.DataFrame:
    diagnosis_base = get_df('diagnosis')
    stays = diagnosis_base[SID]

    df = pd.concat([
        diagnosis_base, 
        diagnosis_base['diagnosisstring'].str.split('|', expand=True)
        ], axis=1)

    masks = {
        "sepsis": ((df[2]=='sepsis') | (df[3]=='due to sepsis') | (df[4]=='sepsis')),
        "pneumonia": df['diagnosisstring'].str.contains('pneumonia'),
        "bronchitis": df['diagnosisstring'].str.contains('bronchitis'),
        "diabetes": df[3].isin(["Type II", "Type I"]),
        "respiratory failure": df['diagnosisstring'].str.contains('respiratory failure'),
        **{d: df['diagnosisstring'].str.contains(d) for d in additional_diagnoses}
    }

    diagnosis = diagnosis_base[[SID]].drop_duplicates()
    for d, mask in masks.items():
        diagnosis[d] = diagnosis[SID].isin(stays[mask])
    return diagnosis.reset_index(drop=True)

def preprocess_treatments(additional_treatments=[], **kwargs) -> pd.DataFrame:
    treatment_base = get_df('treatment')
    stays = treatment_base[SID]

    df = pd.concat([
        treatment_base, 
        treatment_base['treatmentstring'].str.split('|', expand=True)
        ], axis=1)

    additional_treatments += ["ventilation", "insulin", "intravenous fluid", "antibacterials",
                              "penicillins", "vancomycin"]
    masks = {
        **{d: df['treatmentstring'].str.contains(d) for d in additional_treatments}
    }

    treatment = treatment_base[[SID]].drop_duplicates()
    for d, mask in masks.items():
        treatment[d] = treatment[SID].isin(stays[mask])
    return treatment.reset_index(drop=True)

def preprocess_vitalAperiodic(additional_aperiodic={}, **kwargs) -> pd.DataFrame:
    return (
        get_df('vitalAperiodic')
        .groupby(SID)
        .agg({"noninvasivemean": "mean", **additional_aperiodic})
        .rename(columns={"noninvasivemean": "bp"})
        .reset_index()
    )

def preprocess_vitalPeriodic(stays=None, additional_aperiodic={}, **kwargs):
    vitals = {
        "sao2": "mean",
        "heartrate": "mean",
        "respiration": "mean",
        **additional_aperiodic
    }

    vital_chunks = []
    for chunk in tqdm(pd.read_csv(
        "physionet.org/files/eicu-crd/2.0/vitalPeriodic.csv.gz",
        compression="gzip",
        chunksize=1_000_000,
        usecols=[SID] + list(vitals),
        dtype={SID: 'int32', **{vital: 'float32' for vital in vitals}},
        **kwargs
    ), desc="Loading periodic vitals", leave=kwargs.get('verbose', False), total=147):
        if stays is not None:
            chunk = chunk[chunk[SID].isin(stays)]
        vital_chunks.append(chunk)

    return (
        pd.concat(vital_chunks, ignore_index=True)
        .groupby(SID)
        .agg(vitals)
        .reset_index()
    )


PREPROCESS = {
    "patient": preprocess_patient,
    "admissionDx": preprocess_admissionDx,
    "diagnosis": preprocess_diagnoses,
    "treatment": preprocess_treatments,
    "vitalAperiodic": preprocess_vitalAperiodic,
    "vitalPeriodic": preprocess_vitalPeriodic
}

def preprocess(table_filter = None, stay_filter=None, **kwargs) -> pd.DataFrame:
    table_filter = table_filter or PREPROCESS.keys()

    stays = set(get_df('patient')[SID])
    if stay_filter: stays &= set(stay_filter)

    all_dfs = []
    for table, preprocess_table in PREPROCESS.items():
        print(f"Processing {table}...")
        df = preprocess_table(stays=stays, **kwargs)

        if table == 'patient': stays &= set(df[SID])

        all_dfs.append(
            df[df[SID].isin(stays)]
            .set_index(SID)
        )
    return pd.concat(all_dfs, axis=1).reset_index()

def get_preprocessed(path):
    if path.split('.')[-1] != "csv":
        path = f"out/processed_data/{path}.csv"
    return pd.read_csv(path)

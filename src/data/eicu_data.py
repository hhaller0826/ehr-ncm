import pickle
from tqdm import tqdm
from pandas import DataFrame
from abc import ABC, abstractmethod
from src.parser import eICUConfigParser
from src.helpers import *

class eICUData:
    stayID = "patientunitstayid"
    groupID = ["uniquepid", "hospitalid"]

    def __init__(self, config: eICUConfigParser, verbose=False, **kwargs):
        self.config = eICUConfigParser.get_parser(config)
        self.verbose = verbose 

        patient_df = self.config.get_patient_df(**kwargs)
        self.id_map = patient_df[["patientunitstayid", "uniquepid", "hospitalid"]].drop_duplicates()
        self.data = self.parse_data(patient_df)
    
    # vitals, treatments, demographics, diagnoses, features list
    @property 
    def demographics(self): return [self.config.rename_cols.get(item, item) for item in list(self.config.demographics.keys())]
    @property
    def diagnoses(self): return [self.config.rename_cols.get(item, item) for item in to_iter(self.config.diagnosis)]
    @property
    def treatments(self): return [self.config.rename_cols.get(item, item) for item in to_iter(self.config.treatment)]
    @property
    def vitals(self):
        vitals = list(self.config.vitalAperiodic) + list(self.config.vitalPeriodic)
        return [self.config.rename_cols.get(item, item) for item in vitals]
    @property
    def features(self): return list(self.data.columns[2:])    

    def parse_data(self, patient_df) -> DataFrame:
        # initialize with demographic information
        data = self._get_agg(patient_df, self.config.demographic_agg_dict())
        data['gender'] = data['gender'].map(lambda x: x if x in ['Male', 'Female'] else 'Other/Unknown')
        data['ethnicity'] = data['ethnicity'].fillna('Other/Unknown')

        # append diagnoses and treatments to the dataframe
        if self.verbose: print("Getting diagnoses and treatments...")
        data = data.merge(self.get_diagnoses_and_treatments().fillna(0), on=self.groupID)

        # append vitals to the dataframe
        if self.verbose: print("Getting vitals...")
        data = data.merge(self.get_vitals(), on=self.groupID, how="left")

        data = data.rename(columns=self.config.rename_cols or {})
        return data
    
    def merge_id(self, df): return df.merge(self.id_map, on=self.stayID, how="inner")

    def get_diagnoses_and_treatments(self) -> DataFrame:
        diagnoses = self.merge_id(get_df("diagnosis", usecols=[self.stayID, "diagnosisstring"]))
        treatments = self.merge_id(get_df("treatment", usecols=[self.stayID, "treatmentstring"]))

        treatment_map = self.config.treatment_map or {}

        for d in self.config.diagnosis:
            diagnoses[d] = diagnoses["diagnosisstring"].str.contains(d, case=False)
            d_stays = diagnoses.loc[diagnoses[d] == True, self.stayID]

            for t in treatment_map[d]:
                mask = (
                    treatments[self.stayID].isin(d_stays) & treatments["treatmentstring"].str.contains(t)
                )
                if t not in treatments.columns: treatments[t] = False
                treatments.loc[mask, t] = True 
                
        diagnoses = self._get_bin(diagnoses, self.config.diagnosis)
        treatments = self._get_bin(treatments, self.config.treatment)
        return diagnoses.merge(treatments, on=self.groupID, how="left")

    def _get_bin(self, df, bin_list):
        return (
            df
            .groupby(self.groupID)
            [bin_list]
            .any()
            .astype(int)
            .reset_index()
        )

    def get_vitals(self) -> DataFrame:
        # Get aperiodic vitals
        vitalAperiodic = get_df("vitalAperiodic", usecols=[self.stayID] + list(self.config.vitalAperiodic))
        vitalAperiodic = self.merge_id(vitalAperiodic)
        vitalAperiodic = self._get_agg(vitalAperiodic, self.config.vitalAperiodic)

        # Get periodic vitals
        dtype = {self.stayID: 'int32', **{vital: 'float32' for vital in self.config.vitalPeriodic}}
        vitalPeriodic = eICUData.get_vital_periodic_df(
            id_map=self.id_map,
            stays=self.id_map[self.stayID].unique(),
            usecols=list(dtype),
            dtype=dtype,
            verbose=False
        )
        vitalPeriodic = self._get_agg(vitalPeriodic, self.config.vitalPeriodic)
        return vitalAperiodic.merge(vitalPeriodic, on=self.groupID, how="left")

    def _get_agg(self, df, agg_dict):
        return (
            df 
            .groupby(self.groupID)
            .agg(agg_dict)
            .reset_index()
        )

    @staticmethod
    def get_vital_periodic_df(
        id_map,
        stays=None,
        **kwargs
    ):
        vital_chunks = []
        for chunk in tqdm(pd.read_csv(
            "physionet.org/files/eicu-crd/2.0/vitalPeriodic.csv.gz",
            compression="gzip",
            chunksize=1_000_000,
            **kwargs
        ), desc="Loading periodic vitals", leave=kwargs.get('verbose', False), total=147):
            if stays is not None:
                chunk = chunk[chunk['patientunitstayid'].isin(stays)]
            chunk = chunk.merge(
                id_map,
                on="patientunitstayid",
                how="inner"
            )
            vital_chunks.append(chunk)

        return pd.concat(vital_chunks, ignore_index=True)
    
    def save(self, path: str):
        if path.split('.')[-1] != "pkl": 
           path = f"out/processed_data/{path}.pkl"
        with open(path, 'wb') as file:
            pickle.dump(self, file)
        if self.verbose: print(f"Saved to {path}")

    @staticmethod
    def load(path: str):
       if path.split('.')[-1] != "pkl": 
           path = f"out/processed_data/{path}.pkl"
       with open(path, 'rb') as fin:
           obj = pickle.load(fin)
       return obj

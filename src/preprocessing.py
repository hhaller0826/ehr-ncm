import pandas as pd
from tqdm import tqdm

from src.utils import *

class eICUDataSettings:
    agg_tables = ["vitalAperiodic"]
    bin_tables = ["diagnosis", "treatment"]

    def __init__(self, 
                 config,
                 **kwargs):
        settings = load_json(f"configs/{config}.json")

        self.hospital_filter = settings.get("hospitals")
        self.demographics = settings.get("demographics", [])
        self.tables = settings.get("tables", {})

        self.additional_data = {}
        for key, val in kwargs.items():
            self.additional_data[key] = val

    @staticmethod
    def isagg(table: str): return table in eICUDataSettings.agg_tables
    @staticmethod
    def isbin(table: str): return table in eICUDataSettings.bin_tables

    def get_bin_tables(self): 
        return set(self.tables) & set(eICUDataSettings.bin_tables)

class eICUData:
    def __init__(self, config="default", verbose=False):
        self.settings = eICUDataSettings(config)

        patient_df = get_df("patient")
        if self.settings.hospital_filter is not None:
            patient_df = patient_df[patient_df['hospitalid'].isin(self.settings.hospital_filter)]
        patient_df['mortality'] = patient_df['unitdischargestatus'].map({'Expired': 1, 'Alive': 0})

        self.id_map = patient_df[[
            "patientunitstayid",
            "uniquepid",
            "hospitalid"
        ]]

        self.verbose = verbose

        self.data = patient_df[["uniquepid", "hospitalid", "mortality"] + self.settings.demographics]

        # binary tables first so we can fillna(0) before moving on to other types
        for table in self.settings.get_bin_tables():
            self._add_data(self._get_bin(table))
            # if table == "diagnosis":
            #     self._add_data(self._get_diagnoses())
        self.data = self.data.fillna(0)

        for table in self.settings.tables:
            if eICUDataSettings.isbin(table):
                continue
            elif eICUDataSettings.isagg(table):
                data = self._get_agg(table)
                if table == "vitalAperiodic":
                    data = data.rename(columns={'noninvasivemean': "bp"})
                self._add_data(data)

            elif table == "vitalPeriodic":
                self._add_data(self._get_vital_periodic())
        
        
    def _add_data(self, df):
        self.data = self.data.merge(
            df,
            on=["uniquepid", "hospitalid"],
            how="left"
        )
    
    def _get_bin(self, table):
        if self.verbose: print(f"Retrieving {table}...")
        bin_list = self.settings.tables[table]
        df = get_df(table).merge(
            self.id_map,
            on="patientunitstayid",
            how="left"
        )

        for item in bin_list:
            df[item] = df[table + "string"].str.contains(item, case=False)

        return (
            df
            .groupby(["uniquepid", "hospitalid"])
            [bin_list]
            .any()
            .astype(int)
            .reset_index()
        )
    
    def _get_agg(self, table):
        if self.verbose: print(f"Retrieving {table}...")
        agg_dict = self.settings.tables[table]
        df = get_df(table, usecols=["patientunitstayid"] + list(agg_dict.keys())).merge(
            self.id_map,
            on="patientunitstayid",
            how="left"
        )
        return (
            df
            .groupby(["uniquepid", "hospitalid"])
            .agg(agg_dict)
            .reset_index()
        )
        
    def _get_vital_periodic(self):
        """
        Handle vitalPeriodic table specially because it's so long it must be batched to load. 
        """
        stays = self.id_map["patientunitstayid"].unique()
        vital_chunks = []

        for chunk in tqdm(pd.read_csv(
            "physionet.org/files/eicu-crd/2.0/vitalPeriodic.csv.gz",
            compression="gzip",
            chunksize=1_000_000,
            usecols=["patientunitstayid", "sao2", "heartrate", "respiration"],
            dtype={"patientunitstayid": 'int32', "sao2": 'float32', "heartrate": 'float32', "respiration": 'float32'},
        ), desc="Loading periodic vitals", leave=self.verbose, total=147):
            chunk = chunk[chunk['patientunitstayid'].isin(stays)]
            chunk = chunk.merge(
                self.id_map,
                on="patientunitstayid",
                how="left"
            )
            vital_avg = (
                chunk
                .groupby(["uniquepid", "hospitalid"])
                .agg(
                    sao2=("sao2", "mean"),
                    hr=("heartrate", "mean"),
                    respiration=("respiration", "mean"),
                )
                .reset_index()
            )
            vital_chunks.append(vital_avg)

        vitals = pd.concat(vital_chunks, ignore_index=True)
        return vitals
    
from abc import ABC, abstractmethod

from src.helpers import *

class ConfigParser(ABC):
    def __init__(self, config="default", **kwargs):
        self.config = config if isinstance(config, dict) else load_json(f"configs/{config}.json")

    def __getattr__(self, name):
        # if name == "config": return self.config
        # return self.config.get(name, None)
        config = self.__dict__.get("config", None)
        if config is None:
            raise AttributeError(name)
        return config.get(name, None)
    
    @staticmethod
    def get_parser(config):
        if isinstance(config, ConfigParser): return config 
        return ConfigParser(config)

class eICUConfigParser(ConfigParser):
    def get_patient_df(self, hospital_filter = []):
        patient_df = get_df("patient")
        hospital_filter += to_iter(self.hospital_filter)
        if len(hospital_filter) > 0:
            patient_df = patient_df[patient_df['hospitalid'].isin(hospital_filter)]
        patient_df["age"] = pd.to_numeric(patient_df['age'].replace({'> 89': '90'}), errors='coerce')
        patient_df["mortality"] = patient_df['unitdischargestatus'].map({'Expired': 1, 'Alive': 0})
        return patient_df
    
    def demographic_agg_dict(self):
        return {**self.demographics, "mortality": "max"} 

    # info for getting demographics, diagnoses, treatments, vitals 

    # get causal graph from configs
    def _append_additional_info(self, **kwargs):
        return super()._append_additional_info(**kwargs)
    
    @staticmethod
    def get_parser(config):
        if isinstance(config, eICUConfigParser): return config 
        return eICUConfigParser(config)
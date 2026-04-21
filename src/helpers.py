import json
import torch
import pandas as pd

DEVICE = torch.device("mps" if torch.mps.is_available() else "cpu")

def get_df(filename, **kwargs):
    return pd.read_csv("physionet.org/files/eicu-crd/2.0/" + filename + ".csv.gz", compression="gzip", **kwargs)

def load_json(filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print("File not found.")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format.")

def to_iter(val):
    if isinstance(val, (str, bytes)): return [val]
    try:
        iter(val)
        return val
    except TypeError:
        return [val]

def get_hospitals_with(tables='all'):
    if tables=='all': tables = ["diagnosis", "treatment", "vitals"]
    hospital_map = load_json("src/data/hospitals_with_data.json")
    hospitals = [set(hospital_map.get(t, [])) for t in tables]
    return list(set.intersection(*hospitals))
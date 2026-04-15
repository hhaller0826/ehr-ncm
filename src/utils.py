import json
import pandas as pd

DATA_FOLDER = "physionet.org/files/eicu-crd/2.0/"

def get_df(filename, **kwargs):
    return pd.read_csv(DATA_FOLDER + filename + ".csv.gz", compression="gzip", **kwargs)

def load_json(filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print("File not found.")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format.")
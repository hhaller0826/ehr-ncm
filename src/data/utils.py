import warnings
import pandas as pd

from . import eICUData, ProcessedData
from src.helpers import *

warnings.simplefilter(action='ignore', category=FutureWarning)

def process_eicu_data(eicu_dt: eICUData, config=None, graph=None, **kwargs):
    config = config or eicu_dt.config
    data = eicu_dt.data.copy().fillna(0)
    for col, col_bins in config.column_bins.items():
        data[col] = pd.cut(data[col], **col_bins)
    
    check_assignments(data, config.assignments, graph)
    return ProcessedData(
        data,
        assignments = config.assignments,
        categorical_vars = eicu_dt.demographics + eicu_dt.vitals,
        # discrete_vars = ['mortality'] + eicu_dt.diagnoses + eicu_dt.treatments,
        **kwargs
    )

def check_assignments(data, assignments: dict, graph):
    """
    Check for assigning data columns to graph nodes.
    """
    # Check that all nodes are being assigned
    if graph is not None:
        assert assignments.keys() <= graph.set_v, f'Node {assignments.keys()-graph.set_v} not in graph'
    
    assigned_features = []
    for features in assignments.values():
        assert features is not None and len(features) > 0, f'All nodes must have an assignment'
        assigned_features.extend(features)
    feature_set = set(assigned_features)
    # check for duplicate features:
    if len(feature_set) < len(assigned_features):
        seen = set()
        duplicates = {x for x in assigned_features if x in seen or seen.add(x)}
        raise ValueError('Feature was assigned to a variable more than once: {}'.format(duplicates))

    # check for unknown features:
    cols = set(data.columns) # assuming this is a DataFrame object rn 
    unknown_features = feature_set - cols
    if len(unknown_features) > 0:
        raise ValueError('Unknown feature assignment: {}'.format(unknown_features))

    # check for missing features (this is OK):
    unassigned_features = cols - feature_set
    if len(unassigned_features) > 0:
        warnings.warn('The following features were not assigned to any variable: {}'.format(unassigned_features), UserWarning)
        print("It is okay to exclude features from the model but they will not be used in the causal analysis.")

    return feature_set

def get_treatments_for(diagnosis):
    diagnoses = get_df("diagnosis")
    stays = diagnoses[diagnoses["diagnosisstring"].str.contains(diagnosis)]["patientunitstayid"]

    treatments = get_df("treatment")
    treatments = treatments[treatments["patientunitstayid"].isin(stays)]
    return treatments["treatmentstring"]

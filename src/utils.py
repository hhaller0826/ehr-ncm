import pandas as pd 
from src.data import eICUData, process_eicu_data
from src.parser import eICUConfigParser
from src.model import FF_NCM
from src.graph import CausalGraph
from itertools import combinations

def get_eicu_data_bundle(config, datapath=None, modelpath=None, **kwargs):
    config = eICUConfigParser.get_parser(config)
    cg = get_graph_from_config(config)

    eicu_dt = eICUData.load(datapath) if datapath is not None else eICUData(config, **kwargs)
    processed_dt = process_eicu_data(eicu_dt, config=config, graph=cg, **kwargs)
    ncm = FF_NCM.load(
        path=modelpath,
        cg=cg, 
        v_size={k:len(v) for k,v in config.assignments.items()},
        scale=processed_dt.get_assigned_scale(),
        **kwargs
    )

    return eicu_dt, processed_dt, ncm

def get_graph_from_config(config) -> CausalGraph:
    config = eICUConfigParser.get_parser(config)
    nodes = list(config.assignments.keys())
    de = [(p, ch)
          for p in config.causal_dict
          for ch in config.causal_dict[p]]
    be = []
    for group in config.bidirected:
        be += list(combinations(group, 2))

    return CausalGraph(nodes, de, be)
    
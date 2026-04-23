import torch as T
import warnings 
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def process_data_columns(df, columns, categorical_vars=[], continuous_vars=[], **kwargs):
        abbr_df = df.copy()
        scale = {}
        ret_map = {}

        for feat in columns:
            if feat in categorical_vars:
                encoder = LabelEncoder()
                abbr_df[feat] = encoder.fit_transform(abbr_df[feat])

                maxval = abbr_df[feat].nunique()-1
                minval = 0
                ret_map[feat] = [encoder.inverse_transform([i]).item() for i in range(maxval+1)]
            else:
                maxval = abbr_df[feat].max()
                minval = abbr_df[feat].min()
            
            if feat in continuous_vars:
                scale[feat] = (lambda x, maxval=maxval, minval=minval: (x*(maxval-minval)) + minval)
            else:
                scale[feat] = (lambda x, maxval=maxval, minval=minval: T.round((x*(maxval-minval)) + minval))

            # easiest to use NN with a sigmoid so we need to normalize the values between 0 and 1
            # TODO: currently just hoping the real max & min values are in the dataset. if this is grades but everyone got B's and C's, then my algo will never predict A or D
            abbr_df[feat] = abbr_df[feat].apply(lambda x: (x-minval)/(maxval-minval))

        return abbr_df, scale, ret_map


class ProcessedData:
    def __init__(self, df, assignments, **kwargs):
        self._assignments = assignments
        self.columns = sum(assignments.values(), [])

        self.data, self.scale, self.ret_map = process_data_columns(df, self.columns, **kwargs)
        self.hosp_data = {hosp: group for hosp, group in self.data.groupby('hospitalid')[self.columns]}

        # self.train_dataloader = NCMDataset(self.train_df, assignments).get_dataloader(batch_size=batch_size)
        # self.test_dataloader = NCMDataset(self.test_df, assignments).get_dataloader(batch_size=batch_size)
    @property
    def assignments(self): return self._assignments
    
    @assignments.setter
    def assignments(self, new_assignments):
        self._assignments = new_assignments
        self.columns = sum(new_assignments.values(), [])

    def __getitem__(self, col): return self.data[col]

    def train_test_hospital_split(self, train_hospitals=[], test_size=0.1, **kwargs):
        train_dict, test_dict = {}, {}
        for h, df in self.hosp_data.items():
            if train_hospitals=='all' or h in train_hospitals:
                train_dict[h], test_dict[h] = train_test_split(df, test_size=test_size, **kwargs)
            else:
                test_dict[h] = df 
        return train_dict, test_dict
    
    def train_test_split(self, train_hospitals=[], test_size=0.1, batch_size=32, **kwargs):
        train_dict, test_dict = self.train_test_hospital_split(train_hospitals=train_hospitals, test_size=test_size, **kwargs)
        
        train_df, test_df = pd.concat(train_dict.values(), axis=0), pd.concat(test_dict.values(), axis=0)
        
        train_dataloader = NCMDataset(train_df, self.assignments).get_dataloader(batch_size=batch_size, **kwargs)
        test_dataloader = NCMDataset(test_df, self.assignments).get_dataloader(batch_size=batch_size, **kwargs)
        return train_df, test_df, train_dataloader, test_dataloader
    
    def get_assigned_scale(self, assignments=None):
        assignments = assignments if assignments else self.assignments
        return {v: [self.scale[assignments[v][i]] for i in range(len(assignments[v]))] for v in assignments}
    
    def print_df(self, hospitals=[]):
        if len(hospitals) == 0: return self.data[self.columns]
        return pd.concat([self.hosp_data[h] for h in hospitals], axis=0)
    
    def to_cat(self, variable, samples):
        """
        expecting samples = torch.tensor([[sample1],[sample2],...])

        example: my_data.to_cat('A', torch.tensor([[0.0, 1.0],[1.,2.2],[2.2, 0.6]]))
        might return [['African-American', 'Greater than 45'],
                    ['Asian', 'Less than 25'],
                    ['Caucasian', '25 - 45']]
        """
        feature = self.assignments[variable]
        n = range(len(feature))
        return [[self.ret_map[feature[i]][sample[i].int()] for i in n] for sample in samples]
                



class NCMDataset(Dataset):
    def __init__(self, df, assignments):
        self.df = df.reset_index(drop=True)
        self.variables = assignments.keys()
        self.assignments = assignments

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for v in self.variables:
                val = self.df.loc[idx, self.assignments[v]]
                tensor = T.tensor(val, dtype=T.float)
                if tensor.ndim == 0:
                    tensor = tensor.unsqueeze(0)
                sample[v] = tensor
            return sample
    
    def get_dataloader(self, batch_size=32, shuffle=True, **kwargs):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, **kwargs)
    
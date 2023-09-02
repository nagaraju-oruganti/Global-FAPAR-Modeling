import os
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import pickle

### Create folds
def create_folds(df, target):
    
    df['kfold'] = -1
    
    df['target_bins'] = pd.qcut(df[target], q = 4, labels = False)
    X = df.drop(columns = 'target_bins')
    y = df['target_bins']
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for idx, (_, vidx) in enumerate(skf.split(X, y), start = 1):
        df.loc[vidx, 'kfold'] = idx
        
    print(df['kfold'].value_counts())
    df.drop(columns = ['target_bins'], inplace = True)
    return df


### DATA PREPROCESSING
class Preprocess:
    def __init__(self, config):
        self.config = config
        self.mode = config.mode
        
    ### LOAD DATASET
    def load_dataset(self):
        train_path = os.path.join(self.config.data_dir, 'train.csv')
        test_path = os.path.join(self.config.data_dir, 'test.csv')
        self.df = pd.read_csv(train_path if self.mode == 'train' else test_path)
        
    ### FEATURE ENGINEERING
    def feature_engineering(self):
        
        if self.mode == 'train':
            # create fold
            self.df = create_folds(self.df, target = self.config.target)
        
        # Assign group to station
        #groups = self.config.station_groups
        #self.df['station_group'] = self.df['station'].apply(lambda s: 1 if s in groups[1] else (2 if s in groups[2] else 3))
        
        # one-hot-encoding
        ohe_df = pd.concat([pd.get_dummies(self.df[e], prefix=e) for e in self.config.ohe_cols], axis = 1)
        ohe_cols = ohe_df.columns.tolist()
        
        # columns
        self.target   = self.config.target
        self.cat_cols = ohe_cols
        remove_cols = ['sample_id', 'kfold', 'station', 'month'] + ohe_cols + [self.target]
        self.num_cols = [c for c in self.df.columns if c not in remove_cols]
        
        # combine
        self.df = pd.concat([self.df, ohe_df], axis = 1)
        
        # remove cols
        self.df.drop(columns = self.config.drop_cols, inplace = True)
       
    def normalize(self):
        num_cols = self.num_cols
        if self.mode == 'train':
            ss = StandardScaler()
            ss.fit(self.df[num_cols])
            self.df[num_cols] = ss.transform(self.df[num_cols])
            self.save(ss, 'scaler')
        else:
            ss = self.load('scaler')
            self.df[num_cols] = ss.transform(self.df[num_cols])

    def save(self, data, name):
        with open(os.path.join(self.config.fe_dir, f'{name}.pkl'), 'wb') as f:
            pickle.dump(data, f)
        
    def load(self, name):
        with open(os.path.join(self.config.fe_dir, f'{name}.pkl'), 'rb') as f:
            return pickle.load(f)
        
    def run(self):
        os.makedirs(self.config.fe_dir, exist_ok=True)
        dest_path = os.path.join(self.config.fe_dir, f'{self.config.mode}.csv')
        if not os.path.exists(dest_path):
            self.load_dataset()             # load dataset
            self.feature_engineering()      # make feature engineered columns
            self.normalize()                # normalize dataset
            self.df.to_csv(dest_path)
        else:
            self.df = pd.read_csv(dest_path)
            
        print(self.df.shape)
        
if __name__ == '__main__':
    from helper_config import Config
    config = Config()
    config.data_dir = 'inputs/data'
    config.fe_dir   = 'inputs/fe'
    config.mode     = 'test'
    p = Preprocess(config)
    p.run()
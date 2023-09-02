
class Config:
    
    data_dir = '../inputs/data'
    models_dir = '../inputs/models'
    fe_dir = '../inputs/fe'
    
    ## Train or inference
    mode = 'train'              # train or test
    
    ## Preprocess / feature engineering
    station_groups = {
        1 : [6, 7, 8, 24, 25, 26, 37, 38, 39, 43, 44, 45, 49, 50, 102, 103],
        2 : [66, 71, 72, 73, 74],
        3 : [4, 5, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 52, 53, 54, 55, 56, 57, 67, 68, 69]
    }
    
    # one-hot-encoded columns
    ohe_cols    = ['month']                             
    drop_cols   = ['station', 'month']
    target      = 'fapar'
    

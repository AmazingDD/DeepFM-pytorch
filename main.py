import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from deepfm import DeepFM

torch.manual_seed(2022)
data = pd.read_csv('./temp_data.csv').reset_index(drop=True)

category_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
dummy_cols = ['SK_ID_CURR']
target_col = 'TARGET'
numeric_cols = list(set(data.columns) - set(category_cols + dummy_cols + [target_col]))

def data_massage(data,  category_cols, numeric_cols):
    feat_cols = category_cols + numeric_cols
    fields = []
    for feat_col in feat_cols:
        if feat_col not in category_cols:
            fields.append(1)
        else:
            fields.append(data[feat_col].nunique())
    start_idx = [0] + np.cumsum(fields)[:-1].tolist()

    return feat_cols, start_idx, fields

class FMDataset(Dataset):
    def __init__(self, data, feat_start_idx, fields_size, feat_cols, target_col):
        self.data = data
        self.label = np.asarray(self.data[target_col])

        self.feat_cols = feat_cols
        self.fields = fields_size
        self.start_idx = feat_start_idx

    def __getitem__(self, index):
        row = self.data.loc[index, self.feat_cols]
        idxs = list()
        vals = list()
        # label = self.data.loc[index, self.]
        label = self.label[index]
        for i in range(len(row)):
            if self.fields[i] == 1:
                idxs.append(self.start_idx[i])
                vals.append(row[i])
            else:
                idxs.append(int(self.start_idx[i] + row[i]))
                vals.append(1)

        label = torch.tensor(label, dtype=torch.float32)
        idxs = torch.tensor(idxs, dtype=torch.long)
        vals = torch.tensor(vals, dtype=torch.float32)
        
        return label, idxs, vals

    def __len__(self):
        return len(self.data)

feat_cols, feat_start_idx, fields_size = data_massage(data,  category_cols, numeric_cols)

args = {
    'batch_size': 256,
    'gpuid': '0',
    'lr': 0.001,
    'l2_reg': 0.,
    'epochs': 10,
    'num_features': len(feat_cols),
    'embedding_dim': 8,
    'field_size': fields_size,
    'num_layers': 2,
    'dense_size': 32,
    '1o_dropout_p': 1., 
    '2o_dropout_p': 1., 
    'deep_dropout_p': 0.5,
    'batch_norm': True,
    'deep_layer_act': 'relu',
    'opt_name': 'adam'
}

train_data, test_data = train_test_split(data, test_size=0.2)
train_data, test_data = train_data.reset_index(drop=True), test_data.reset_index(drop=True)

train_dataset = FMDataset(train_data, feat_start_idx, fields_size, feat_cols, target_col)
train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)

test_dataset = FMDataset(test_data, feat_start_idx, fields_size, feat_cols, target_col)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

model = DeepFM(args)
model.fit(train_loader)
model.predict(test_loader)



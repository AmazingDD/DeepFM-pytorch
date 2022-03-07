# DeepFM-pytorch
re-implementation of DeepFM with pytorch 1.0

# Usage
## Input Format for fit method in model
This implementation requires model to receive data batches in the following format:
- [ ] **label**: target of each sample in the dataset (1/0 for classification)
- [ ] **idxs**: *[[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]*
    - *indi_j* is the feature index of feature field *j* of sample *i* in the dataset
- [ ] **vals**: *[[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]*
    - *vali_j* is the feature value of feature field *j* of sample *i* in the dataset
    - *vali_j* can be either binary (1/0, for binary/categorical features) or float (e.g., 12.34, for numerical features)

## how to run

The `main.py` has already provided methods to process numeric and categorical features, so the only thing you need to do is changing `data_path` and hard coding the column names to tell the program which columns you want to re-format.

e.g. 
 - `dummy_cols` is the list of template codes to show result index; 
 - `category_cols` is the list of categorical column names
 - Confirm that there is no other columns except columns mentioned above in your own dataframe, then the code will automatically extract the `numeric_cols`
```
data = pd.read_csv('./temp_data.csv').reset_index(drop=True)
category_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
dummy_cols = ['SK_ID_CURR']
target_col = 'TARGET'
numeric_cols = list(set(data.columns) - set(category_cols + dummy_cols + [target_col]))
```

# Reference

*DeepFM: A Factorization-Machine based Neural Network for CTR Prediction*, Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.

# import os
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# def load_client_data(cid: int, task_name: str = "interval"):
#     # if task_name == "interval":
#     return load_interval_data_by_id(cid)


# hl_features = [
#     "WBC_10_9_L", "RBC_10_12_L", "HGB_g_L", "HCT_PCT",
#     "MCV_fL", "MCH_pg", "MCHC_g_dL", "PLT_10_9_L",
#     "RDW_SD_fL", "NEUT_10_9_L", "LYMPH_10_9_L",
#     "MONO_10_9_L", "EO_10_9_L", "BASO_10_9_L",
#     "NRBC_10_9_L", "IG_10_9_L"
# ]


# class INTERVALDataset(Dataset):
#     def __init__(self, features, labels):
#         self.features = torch.FloatTensor(features)
#         self.labels = torch.LongTensor(labels)

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         return self.features[idx], self.labels[idx]

# def load_interval_data_by_id(cid: int, batch_size=128):
#     base_path = "/app/datasets/interval/INTERVAL_by_fl_site_v2" 

#     files = [
#         {"train": "INTERVAL_irondef_Non-European site_train.csv", "val": "INTERVAL_irondef_Non-European site_val.csv"},
#         {"train": "INTERVAL_irondef_EUR old site_train.csv", "val": "INTERVAL_irondef_EUR old site_val.csv"},
#         {"train": "INTERVAL_irondef_EUR young site_train.csv", "val": "INTERVAL_irondef_EUR young site_val.csv"},
#     ]
    
#     if cid >= len(files):
#         raise ValueError(f"Client ID {cid} is out of bounds for file list.")

#     train_file = f"{base_path}/{files[cid]['train']}"
#     val_file = f"{base_path}/{files[cid]['val']}"
    
#     try:
#         train_df = pd.read_csv(train_file)
#         val_df = pd.read_csv(val_file)
#     except FileNotFoundError:
#         print("="*50)
#         print(f"ERROR: Data files not found for client {cid} at: {base_path}")
#         print("Please update the `base_path` in pytorchexample/client_app.py")
#         print("="*50)
#         raise

#     train_features = hl_features + ["Age", "Sex"]
#     train_df['Sex'] = (train_df['Sex'] == 'M').astype(np.float32)
#     val_df['Sex'] = (val_df['Sex'] == 'M').astype(np.float32)

#     for col in train_features:
#         if col != 'Sex':
#             train_df[col] = train_df[col].astype(np.float32)
#             val_df[col] = val_df[col].astype(np.float32)

#     X_train = train_df[train_features].values
#     y_train = train_df["ferritin_low"].values.astype(np.int64)
#     X_val = val_df[train_features].values
#     y_val = val_df["ferritin_low"].values.astype(np.int64)


#     mean = X_train.mean(axis=0)
#     std = X_train.std(axis=0)
#     X_train = (X_train - mean) / (std + 1e-8)
#     X_val = (X_val - mean) / (std + 1e-8)

#     train_dataset = INTERVALDataset(X_train, y_train)
#     val_dataset = INTERVALDataset(X_val, y_val)

 
#     class_counts = np.bincount(y_train)
#     class_weights = 1.0 / (class_counts + 1e-6)
#     sample_weights = class_weights[y_train]

#     train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, val_loader




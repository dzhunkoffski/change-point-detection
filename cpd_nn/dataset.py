import numpy as np
from typing import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import math
import torch
import pandas as pd
import pandas as pd
from synth_ts.synth_timeserie_generator import *

class SynthDatasetClassification:
    def __init__(self, ts: np.array, min_stability_duration: int, window_size: int, 
                train_size: float, horizon: int, feature_type: str,
                balancing: str, pos_ratio: float, lap: int = 0, is_round=False, round_decimals=0):
        self.ts = ts
        cp_ixs = np.array(
            find_change_points(ts, min_stability_duration)
        )
        # cp_ixs[i] = 1 => new stability starting from i+1
        self.cp_ratio = len(cp_ixs) / len(ts)

        self.labels = np.zeros(len(self.ts))
        self.labels[cp_ixs] = 1
        self.balancing = balancing
        self.feature_type = feature_type
        self.window_size = window_size
        self.lag = lap
        self.horizon = horizon

        self.x, self.y = [], []

        if is_round:
            self.ts = np.around(self.ts, decimals=round_decimals)
        for i in range(len(self.ts) - window_size - horizon):
            if feature_type == 'absolute':
                self.x.append(self.ts[i : i + window_size]) # window: [i, .. i-1]
                self.y.append(
                    min(len(np.where(
                        self.labels[i + window_size - lap: i + window_size + horizon - lap] == 1
                    )[0]), 1)
                )
            elif feature_type == 'binary':
                self.x.append(self.labels[i : i + window_size])
                self.y.append(
                    min(len(np.where(
                        self.labels[i + window_size - lap : i + window_size + horizon - lap] == 1
                    )[0]), 1)
                )
            elif feature_type == 'stationary':
                if i == 0:
                    divider = self.ts[i : i + window_size - 1]
                    divider[np.where(divider == 0.0)] = 1
                    interval = self.ts[i : i + window_size]
                    interval[1 : window_size] -= self.ts[i : i + window_size - 1]
                    interval[1 : window_size] /= divider
                    self.x.append(
                        interval
                    )
                else:
                    divider = self.ts[i - 1 : i + window_size - 1]
                    divider[np.where(divider == 0.0)] = 1
                    self.x.append(
                        (self.ts[i : i + window_size] - self.ts[i - 1 : i + window_size - 1]) / divider
                    )
                self.y.append(
                    min(len(np.where(
                        self.labels[i + window_size - lap : i + window_size + horizon - lap] == 1
                    )[0]), 1)
                )
        self.x = np.array(self.x).astype(np.float32)
        self.y = np.array(self.y).astype(np.float32)

        # Get Splits
        self.threshold = round(train_size * len(self.x))
        self.pos_ixs = np.argwhere(self.y == 1)
        self.pos_ixs = self.pos_ixs[np.where(self.pos_ixs < self.threshold)]
        self.neg_ixs = np.argwhere(self.y == 0)
        self.neg_ixs = self.neg_ixs[np.where(self.neg_ixs < self.threshold)]
        self.ixs = []

        if balancing == 'undersampling':
            cnt_use_neg_ixs = round(len(self.pos_ixs) / pos_ratio - len(self.pos_ixs))
            use_neg_ixs = np.random.choice(self.neg_ixs, cnt_use_neg_ixs, replace=False)
            self.ixs = np.concatenate((self.pos_ixs, use_neg_ixs))
        elif balancing == 'oversampling':
            neg_ratio = 1 - pos_ratio
            cnt_use_pos_ixs = round(len(self.neg_ixs) / neg_ratio - len(self.neg_ixs))
            use_pos_ixs = np.random.choice(self.pos_ixs, cnt_use_pos_ixs, replace=True)
            self.ixs = np.concatenate((self.neg_ixs, use_pos_ixs))
    
    def get_trainloader(self, batch_size, model_type, shuffle_train=True):
        if self.balancing  == 'None':
            if model_type == 'LSTM':
                x = torch.tensor(self.x[:self.threshold]).unsqueeze(dim=-1)
            elif model_type == 'MLP':
                x = torch.tensor(self.x[:self.threshold])

            return DataLoader(
                torch.utils.data.TensorDataset(
                    x,
                    torch.tensor(self.y[:self.threshold]).unsqueeze(dim=-1)
                ), shuffle=shuffle_train, batch_size=batch_size
            )
        else:
            if model_type == 'LSTM':
                x = torch.tensor(self.x[self.ixs]).unsqueeze(dim=-1)
            elif model_type == 'MLP':
                x = torch.tensor(self.x[self.ixs])
            return DataLoader(
                torch.utils.data.TensorDataset(
                    x,
                    torch.tensor(self.y[self.ixs]).unsqueeze(dim=-1)
                ), shuffle=shuffle_train, batch_size=batch_size
            )
    
    def get_testloader(self, batch_size, model_type):
        if model_type == 'LSTM':
            x = torch.tensor(self.x[self.threshold:]).unsqueeze(dim=-1)
        elif model_type == 'MLP':
            x = torch.tensor(self.x[self.threshold:])

        return DataLoader(
            torch.utils.data.TensorDataset(
                x,
                torch.tensor(self.y[self.threshold:]).unsqueeze(dim=-1)
            ), shuffle=False, batch_size=batch_size
        )

class SynthDatasetTimeToCp:
    def __init__(self, ts: np.array, min_stability_duration: int, window_size: int, 
                train_size: float, feature_type: str,
                balancing: str, pos_ratio: float, cooldown: int = 0):
        self.ts = ts
        cp_ixs = np.array(
            find_change_points(ts, min_stability_duration)
        )
        # cp_ixs[i] = 1 => new stability starting from i+1
        self.cp_ratio = len(cp_ixs) / len(ts)

        self.labels = np.zeros(len(self.ts))
        self.labels[cp_ixs] = 1
        self.balancing = balancing
        self.feature_type = feature_type
        self.window_size = window_size
        self.cooldown = cooldown

        self.x, self.y = [], []
        cooldown_counter = 0
        for i in tqdm(range(window_size, len(ts) - window_size // 2)):
            if cooldown_counter > 0:
                cooldown_counter -= 1
                continue
            # choose first nearest change point index
            cps_pos = cp_ixs[np.argwhere(cp_ixs >= i-1)]
            if len(cps_pos) == 0:
                break
            future_cp_ix = cps_pos[0] + 1
            time_to_cp = future_cp_ix - i

            if feature_type == 'absolute':
                self.x.append(self.ts[i - window_size : i])
            elif feature_type == 'binary':
                self.x.append(self.labels[i - window_size : i])
            elif feature_type == 'stationary':
                if i == window_size:
                    interval = self.ts[i - window_size : i]
                    interval[1 : window_size] -= self.ts[i - window_size : i - 1]
                    interval[1 : window_size] /= self.ts[i - window_size : i - 1]
                    self.x.append(interval)
                else:
                    self.x.append(
                        (self.ts[i - window_size : i] - self.ts[i - window_size - 1 : i - 1]) / (self.ts[i - window_size - 1 : i - 1])
                    )
            self.y.append(time_to_cp[0])
            if time_to_cp[0] == 0:
                cooldown_counter = self.cooldown
        self.x = np.array(self.x).astype(np.float32)
        self.y = np.array(self.y).astype(np.float32)
        self.threshold = int(len(self.x) * train_size)

    
    def get_trainloader(self, batch_size, model_type, shuffle_train=True):
        x = torch.tensor(self.x[:self.threshold]).unsqueeze(dim=-1)
        y = torch.tensor(self.y[:self.threshold]).unsqueeze(dim=-1)

        return DataLoader(
            torch.utils.data.TensorDataset(
                x, y
            ), shuffle=shuffle_train, batch_size=batch_size
        )
    
    def get_testloader(self, batch_size, model_type):
        x = torch.tensor(self.x[self.threshold:]).unsqueeze(dim=-1)
        y = torch.tensor(self.y[self.threshold:]).unsqueeze(dim=-1)

        return DataLoader(
            torch.utils.data.TensorDataset(
                x, y
            ), shuffle=False, batch_size=batch_size
        )
    
class MultiDimTs:
    def __init__(self, ts: np.array, min_stability_duration: int, window_size: int, 
                 train_size: float, balancing: str, pos_ratio: float, horizon: int, lag: int = 0, 
                 is_round=False, round_decimals=0):
       self.ts = ts
       cp_ixs = np.array(
           find_change_points(ts, min_stability_duration)
       ) 
       self.cp_ratio = len(cp_ixs) / len(ts)
       self.labels = np.zeros(len(self.ts))
       self.labels[cp_ixs] = 1
       self.balancing = balancing
       self.window_size = window_size
       self.lag = lag
       self.horizon = horizon
       
       self.x = [], self.y = []

       for i in range(len(self.ts) - window_size - horizon):
           absolute_ts = self.ts[i : i + window_size]
           if is_round:
               absolute_ts = np.around(absolute_ts, decimals=round_decimals)
           stationary_ts = []
           if i == 0:
                divider = self.ts[i : i + window_size - 1]
                interval = self.ts[i : i + window_size]
                interval[1 : window_size] -= self.ts[i : i + window_size - 1]
                interval[1 : window_size] /= divider
           else:
               divider = self.ts[i - 1 : i + window_size - 1]
               stationary_ts = (self.ts[i : i + window_size] - self.ts[i - 1 : i + window_size - 1]) / divider
           levels_ts = []
           cp_cnt = 0
           for label in self.labels[i : i + window_size]:
               levels_ts.append(cp_cnt)
               if label == 1:
                   cp_cnt += 1
           levels_ts = np.array(levels_ts)
           


       
           



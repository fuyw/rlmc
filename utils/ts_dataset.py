import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from utils.timefeatures import time_features


MONTH_IN_HOURS   = 30 * 24      # hours
MONTH_IN_MINUTES = 30 * 24 * 4  # 15 minutes


def get_borders(shift_ratio, train_ratio, test_ratio, total_len, seq_len):
    """Get trianing/valid/test data splits."""
    shift_len = int(np.ceil(total_len * shift_ratio))
    df_len = total_len - shift_len
    train_len = int(np.ceil(df_len * train_ratio))
    test_len = int(np.ceil(df_len * test_ratio))
    borders = [
        (shift_len                      , shift_len + train_len),
        (shift_len + train_len - seq_len, total_len - test_len - seq_len),
        (total_len - test_len  - seq_len, total_len)
    ]
    return borders


class Dataset_ETT_minute(Dataset):
    def __init__(self,
                 fdir='dataset/ETT',
                 fname='ETTm1.csv',
                 size=None,
                 flag='train',
                 freq='h',
                 target='OT',
                 features='M',
                 time_feature='timeF',
                 scale=True,
                 ratios=None):

        # time series input/output size
        if size == None:
            self.seq_len   = 24 * 4 * 4    # 4 days
            self.label_len = 24 * 4        # 1 day
            self.pred_len  = 24 * 4        # 1 day
        else:                              # (96, 48, 24)
            self.seq_len, self.label_len, self.pred_len = size

        # train/test/validation dataset
        assert flag in ['train', 'test', 'valid']
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[flag]

        # other settings
        self.freq         = freq
        self.target       = target
        self.features     = features
        self.time_feature = time_feature
        self.scale        = scale

        # read csv file
        self.fdir  = fdir
        self.fname = fname
        self.__read_data__(ratios)

    def __read_data__(self, ratios=None):
        """dataset/ETT/ETTm1.csv ~ 15-minute data
                              date    HUFL   HULL   MUFL   MULL   LUFL   LULL         OT
        0      2016-07-01 00:00:00   5.827  2.009  1.599  0.462  4.203  1.340  30.531000
        1      2016-07-01 00:15:00   5.760  2.076  1.492  0.426  4.264  1.401  30.459999
        2      2016-07-01 00:30:00   5.760  1.942  1.492  0.391  4.234  1.310  30.038000
        3      2016-07-01 00:45:00   5.760  1.942  1.492  0.426  4.234  1.310  27.013000
        """
        self.scaler = StandardScaler()
        raw_df = pd.read_csv(f'{self.fdir}/{self.fname}')  # (69680, 8)

        if ratios is not None:
            shift_ratio, train_ratio, test_ratio = ratios
            borders = get_borders(shift_ratio,
                                  train_ratio,
                                  test_ratio,
                                  len(raw_df),
                                  self.seq_len)
        else:
            borders = [
                (0                                   , 12 * MONTH_IN_MINUTES),    # 12 months 
                (12 * MONTH_IN_MINUTES - self.seq_len, 16 * MONTH_IN_MINUTES),    #  4 months
                (16 * MONTH_IN_MINUTES - self.seq_len, 20 * MONTH_IN_MINUTES),    #  4 months
            ]
        border1, border2 = borders[self.set_type]              # (0, 34560)

        # multivariate features
        if self.features == 'M' or self.features == 'MS':
            raw_data = raw_df.iloc[:, 1:]                      # (69680, 7)
        elif self.features == 'S':
            raw_data = raw_df[[self.target]]

        # scaling data w.r.t. training data
        if self.scale:
            training_data = raw_data[:borders[0][1]]
            self.scaler.fit(training_data.values)
            data = self.scaler.transform(raw_data.values)
        else:
            data = raw_data.values

        # process timestamps
        df_timestamp = raw_df[['date']][border1:border2]
        df_timestamp['date'] = pd.to_datetime(df_timestamp.date)
        if self.time_feature == 'timeF':
            timestamp = time_features(pd.to_datetime(df_timestamp['date'].values), freq=self.freq)
            timestamp = timestamp.transpose(1, 0)
        else:
            df_timestamp['month'] = df_timestamp.date.apply(lambda row: row.month)
            df_timestamp['day'] = df_timestamp.date.apply(lambda row: row.day)
            df_timestamp['weekday'] = df_timestamp.date.apply(lambda row: row.weekday())
            df_timestamp['hour'] = df_timestamp.date.apply(lambda row: row.hour)
            df_timestamp['minute'] = df_timestamp.date.apply(lambda row: row.minute // 15)
            timestamp = df_timestamp.drop(['date'], axis=1).values

        self.data_x = self.data_y = data[border1:border2]
        self.timestamp = timestamp

    def __getitem__(self, index):
        """
        x_start                  x_end 
        |--------------------------|        ==> seq_len
                           |-------|----|   ==> label_len & pred_len
                        y_start       y_end
        """
        x_start_idx   = index
        x_end_idx     = index + self.seq_len
        y_start_idx   = x_end_idx - self.label_len
        y_end_idx     = x_end_idx + self.pred_len

        seq_x = self.data_x[x_start_idx:x_end_idx]
        seq_y = self.data_y[y_start_idx:y_end_idx]
        seq_x_timestamp = self.timestamp[x_start_idx:x_end_idx]
        seq_y_timestamp = self.timestamp[y_start_idx:y_end_idx]

        return seq_x, seq_y, seq_x_timestamp, seq_y_timestamp

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour(Dataset):
    def __init__(self,
                 fdir='dataset/ETT-small',
                 fname='ETTh1.csv',
                 size=None,
                 flag='train',
                 freq='h',
                 target='OT',
                 features='M',
                 time_feature='timeF',
                 scale=True,
                 ratios=None):

        # time series input/output size
        if size == None:
            self.seq_len   = 24 * 4 * 4    # 4 days
            self.label_len = 24 * 4        # 1 day
            self.pred_len  = 24 * 4        # 1 day
        else:                              # (96, 48, 24)
            self.seq_len, self.label_len, self.pred_len = size

        # train/test/validation dataset
        assert flag in ['train', 'valid', 'test']
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[flag]

        # other settings
        self.freq         = freq
        self.target       = target
        self.features     = features
        self.time_feature = time_feature
        self.scale        = scale

        # read csv file
        self.fdir  = fdir
        self.fname = fname
        self.__read_data__(ratios)

    def __read_data__(self, ratios=None):
        """dataset/ETT/ETTh1.csv ~ hourly data
                          date   HUFL   HULL   MUFL   MULL   LUFL   LULL         OT
        0  2016-07-01 00:00:00  5.827  2.009  1.599  0.462  4.203  1.340  30.531000
        1  2016-07-01 01:00:00  5.693  2.076  1.492  0.426  4.142  1.371  27.787001
        2  2016-07-01 02:00:00  5.157  1.741  1.279  0.355  3.777  1.218  27.787001
        3  2016-07-01 03:00:00  5.090  1.942  1.279  0.391  3.807  1.279  25.044001
        4  2016-07-01 04:00:00  5.358  1.942  1.492  0.462  3.868  1.279  21.948000
        """
        self.scaler = StandardScaler()
        raw_df = pd.read_csv(f'{self.fdir}/{self.fname}')  # (17420, 8)

        if ratios is not None:
            shift_ratio, train_ratio, test_ratio = ratios
            borders = get_borders(shift_ratio,
                                  train_ratio,
                                  test_ratio,
                                  len(raw_df),
                                  self.seq_len)
        else:
            borders = [ 
                (0                                 , 12 * MONTH_IN_HOURS),           # 12 months 
                (12 * MONTH_IN_HOURS - self.seq_len, 16 * MONTH_IN_HOURS),           #  4 months
                (16 * MONTH_IN_HOURS - self.seq_len, 20 * MONTH_IN_HOURS),           #  4 months
            ]

        # data = raw_data[border1: border2]
        border1, border2 = borders[self.set_type]               # (0, 34560)

        # multivariate features
        if self.features == 'M' or self.features == 'MS':
            raw_data = raw_df.iloc[:, 1:]                       # (69680, 7)
        elif self.features == 'S':
            raw_data = raw_df[[self.target]]

        # scaling data w.r.t. training data
        if self.scale:
            training_data = raw_data[:borders[0][1]]
            self.scaler.fit(training_data.values)
            data = self.scaler.transform(raw_data.values)
        else:
            data = raw_data.values

        # process time stamps
        df_timestamp = raw_df[['date']][border1:border2]
        df_timestamp['date'] = pd.to_datetime(df_timestamp.date)
        if self.time_feature == 'timeF':
            timestamp = time_features(pd.to_datetime(df_timestamp['date'].values), freq=self.freq)
            timestamp = timestamp.transpose(1, 0)
        else:
            df_timestamp['month'] = df_timestamp.date.apply(lambda row: row.month)
            df_timestamp['day'] = df_timestamp.date.apply(lambda row: row.day)
            df_timestamp['weekday'] = df_timestamp.date.apply(lambda row: row.weekday())
            df_timestamp['hour'] = df_timestamp.date.apply(lambda row: row.hour)
            timestamp = df_timestamp.drop(['date'], axis=1).values

        self.data_x = self.data_y = data[border1:border2]
        self.timestamp = timestamp

    def __getitem__(self, index):
        """
        x_start                  x_end 
        |--------------------------|        ==> seq_len
                           |-------|----|   ==> label_len & pred_len
                        y_start       y_end
        """
        x_start_idx   = index
        x_end_idx     = index + self.seq_len
        y_start_idx   = x_end_idx - self.label_len
        y_end_idx     = x_end_idx + self.pred_len

        seq_x = self.data_x[x_start_idx:x_end_idx]
        seq_y = self.data_y[y_start_idx:y_end_idx]
        seq_x_timestamp = self.timestamp[x_start_idx:x_end_idx]
        seq_y_timestamp = self.timestamp[y_start_idx:y_end_idx]

        return seq_x, seq_y, seq_x_timestamp, seq_y_timestamp

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self,
                 fdir='dataset/ETT-small',
                 fname='ETTh1.csv',
                 size=None,
                 flag='train',
                 freq='h',
                 target='OT',
                 features='S',
                 time_feature='timeF',
                 scale=True,
                 ratios=None):

        # time series input/output size
        if size == None:
            self.seq_len   = 24 * 4 * 4  # 16 Days
            self.label_len = 24 * 4      #  4 Days
            self.pred_len  = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        # train/test/validation dataset
        assert flag in ['train', 'test', 'valid']
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[flag]

        # other settings
        self.freq         = freq
        self.target       = target
        self.features     = features
        self.time_feature = time_feature
        self.scale        = scale

        # read csv file
        self.fdir  = fdir
        self.fname = fname
        self.__read_data__(ratios)

    def __read_data__(self, ratios=None):
        self.scaler = StandardScaler()
        raw_df = pd.read_csv(f'{self.fdir}/{self.fname}')
        cols = [i for i in raw_df.columns if i not in ['date', self.target]]
        raw_df = raw_df[['date', *cols, self.target]]

        # data split
        if ratios is not None:
            shift_ratio, train_ratio, test_ratio = ratios
            borders = get_borders(shift_ratio,
                                  train_ratio,
                                  test_ratio,
                                  len(raw_df),
                                  self.seq_len)
        else:
            num_train = int(len(raw_df) * 0.7)
            num_test  = int(len(raw_df) * 0.2)
            num_valid = len(raw_df) - num_train - num_test 
            borders = [
                (0                                , num_train),
                (num_train-self.seq_len           , num_train+num_valid),
                (len(raw_df)-num_test-self.seq_len, len(raw_df))
            ]

        border1, border2 = borders[self.set_type]

        # multivariate features
        if self.features == 'M' or self.features == 'MS':
            raw_data = raw_df.iloc[:, 1:]
        elif self.features == 'S':
            raw_data = raw_df[[self.target]]

        # scaling data w.r.t. training data
        if self.scale:
            training_data = raw_data[:borders[0][1]]
            self.scaler.fit(training_data.values)
            data = self.scaler.transform(raw_data.values)
        else:
            data = raw_data.values

        # process time stamps
        df_timestamp = raw_df[['date']][border1:border2]
        df_timestamp['date'] = pd.to_datetime(df_timestamp.date) 
        if self.time_feature == 'timeF':
            timestamp = time_features(pd.to_datetime(df_timestamp['date'].values), freq=self.freq)
            timestamp = timestamp.transpose(1, 0)
        else:
            df_timestamp['month'] = df_timestamp.date.apply(lambda row: row.month)
            df_timestamp['day'] = df_timestamp.date.apply(lambda row: row.day)
            df_timestamp['weekday'] = df_timestamp.date.apply(lambda row: row.weekday())
            df_timestamp['hour'] = df_timestamp.date.apply(lambda row: row.hour)
            timestamp = df_timestamp.drop(['date'], axis=1).values

        self.data_x = self.data_y = data[border1:border2]
        self.timestamp = timestamp

    def __getitem__(self, index):
        """
        x_start                  x_end 
        |--------------------------|        ==> encoder: seq_len
                           |-------|----|   ==> decoder: label_len & pred_len
                        y_start       y_end
        """
        x_start_idx   = index
        x_end_idx     = index + self.seq_len
        y_start_idx   = x_end_idx - self.label_len
        y_end_idx     = x_end_idx + self.pred_len

        seq_x = self.data_x[x_start_idx:x_end_idx]
        seq_y = self.data_y[y_start_idx:y_end_idx]
        seq_x_timestamp = self.timestamp[x_start_idx:x_end_idx]
        seq_y_timestamp = self.timestamp[y_start_idx:y_end_idx]

        return seq_x, seq_y, seq_x_timestamp, seq_y_timestamp

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

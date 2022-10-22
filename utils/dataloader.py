from torch.utils.data import DataLoader
from utils.ts_dataset import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom

dataset_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    "electricity": Dataset_Custom,
    "exchange_rate": Dataset_Custom,
    "traffic": Dataset_Custom,
    "weather": Dataset_Custom,
    "illness": Dataset_Custom,
}


def get_data(args, flag, ratios=None, shuffle=True, drop_last=False, scale=True):
    Dataset = dataset_dict[args.dataset]
    batch_size = args.batch_size

    dataset = Dataset(fdir=args.fdir,
                      fname=args.fname,
                      flag=flag,
                      size=[args.seq_len, args.label_len, args.pred_len],
                      features=args.features,
                      target=args.target,
                      time_feature=args.embed,
                      freq=args.freq,
                      ratios=ratios,
                      scale=scale)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=args.num_workers,
                             drop_last=drop_last)
    return data_loader

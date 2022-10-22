"""Format raw csv data to npz files for model inputs: (Xs, Ys, X_ts, Y_ts).
"""
import os
import numpy as np

import torch
import torch.nn.functional as F

from models import Autoformer, ETSformer, FEDformer, Informer, Transformer
from utils.dataloader import get_data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#######################
# Experiment Settings #
#######################
MODEL_DICT = {"Autoformer": Autoformer,
              "ETSformer": ETSformer,
              "FEDformer": FEDformer,
              "Informer": Informer,
              "Transformer": Transformer}
FDIR_DICT = {"ETTh1": ("./dataset/ETT", 'ETTh1.csv'),
             "ETTh2": ("./dataset/ETT", 'ETTh2.csv'),
             "ETTm1": ("./dataset/ETT", 'ETTm1.csv'),
             "ETTm2": ("./dataset/ETT", 'ETTm2.csv'),
             "electricity": ("./dataset/electricity", "electricity.csv"),
             "traffic": ("./dataset/traffic", "traffic.csv"),
             "illness": ("./dataset/illness", "national_illness.csv"),
             "weather": ("./dataset/weather", "weather.csv"),
             "exchange_rate": ("./dataset/exchange_rate", "exchange_rate.csv")}
DIMS_DICT = {"ETTh1": 7,
             "ETTh2": 7,
             "ETTm1": 7,
             "ETTm2": 7,
             "electricity": 321,
             "traffic": 862,
             "illness": 7,
             "weather": 21,
             "exchange_rate": 8}


def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    # basic config
    parser.add_argument('--model', type=str, default='Autoformer')
    parser.add_argument('--seed', type=int, default=2021)

    # data loader setting
    parser.add_argument('--dataset', type=str, default='ETTh1')
    parser.add_argument('--fdir', type=str, default='./dataset/ETT')
    parser.add_argument('--fname', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate/multivariate, S:univariate/univariate, MS:multivariate/univariate')
    parser.add_argument('--target', type=str, default='OT',
                        help='target feature in S or MS task')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h',
                        help='[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='data loader num workers')

    # Autoformer config
    parser.add_argument('--wavelet', type=int, default=0,
                        help='whether use wavelet in Autoformer')

    # ETSformer config
    parser.add_argument('--K', type=int, default=3,
                        help='top-K freq in Fourier layer')
    parser.add_argument('--std', type=float, default=0.2)

    # DLinear config
    parser.add_argument('--individual', action='store_true',
                        default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # FEDformer config
    parser.add_argument('--version', type=str, default='Fourier',
                        help='options: [Fourier, Wavelets]')
    parser.add_argument('--mode_select', type=str, default='random', 
                        help='options: [random, low]')
    parser.add_argument('--modes', type=int, default=64,
                        help='modes to be selected random 64')
    parser.add_argument('--L', type=int, default=3, help='ignore level')
    parser.add_argument('--base', type=str, default='legendre',
                        help='mwt base')
    parser.add_argument('--cross_activation', type=str, default='tanh',
                        help='mwt cross atention activation function tanh or softmax')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=24)

    # Formers 
    parser.add_argument('--e_layers', type=int, default=2,
                        help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1,
                        help='num of decoder layers')
    parser.add_argument('--enc_in', type=int, default=7,
                        help='encoder input size')
    # DLinear with --individual, use this as the number of channels
    parser.add_argument('--dec_in', type=int, default=7,
                        help='decoder input size')
    parser.add_argument('--d_model', type=int, default=512,
                        help='dimension of model')
    parser.add_argument('--c_out', type=int, default=7,
                        help='output size')
    parser.add_argument('--d_ff', type=int, default=2048,
                        help='dimension of fcn')
    parser.add_argument('--n_heads', type=int, default=8)

    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--embed_type', type=int, default=0,
                        help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--moving_avg', type=int, default=25,
                        help='window size of moving average')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true',
                        help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true',
                        help='whether to predict unseen future data')

    # optimization 
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-4, help='lr')
    parser.add_argument('--lradj', type=str, default='type1',
                        help='adjust learning rate')

    args = parser.parse_args()
    return args


def evaluate(args, model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_timestamp, batch_y_timestamp) in enumerate(data_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_timestamp = batch_x_timestamp.float().to(device)
            batch_y_timestamp = batch_y_timestamp.float().to(device)

            # decoder input
            zeros_input = torch.zeros_like(batch_y[:, -args.pred_len:, :])
            decoder_input = torch.cat([batch_y[:, :args.label_len, :], zeros_input], dim=1).to(device)

            # encoder - decoder
            if "DLinear" in args.model:
                outputs = model(batch_x)
            else:
                if args.output_attention:
                    outputs = model(batch_x,
                                    batch_x_timestamp,
                                    decoder_input,
                                    batch_y_timestamp)[0]
                else:
                    outputs = model(batch_x,
                                    batch_x_timestamp,
                                    decoder_input,
                                    batch_y_timestamp)
            feat_dim = -1 if args.features == "MS" else 0
            outputs = outputs[:, -args.pred_len:, feat_dim:]
            batch_y = batch_y[:, -args.pred_len:, feat_dim:]
            loss = F.mse_loss(outputs, batch_y)
            total_loss += loss.item()
    avg_loss = total_loss / (i + 1)
    return avg_loss


def get_npz_datasets(args):
    # load dataset
    shift_ratio, train_ratio, test_ratio = [0.5, 0.7, 0.2]
    ratios = [shift_ratio, train_ratio, test_ratio]

    for flag in ["train", "valid", "test"]:
        # load raw data without scaling
        data_loader = get_data(args, flag=flag, ratios=ratios, shuffle=False, scale=False)

        # load data from the data loader
        Xs, Ys, X_ts, Y_ts = [], [], [], []
        for batch_x, batch_y, batch_x_timestamp, batch_y_timestamp in data_loader:
            Xs.append(batch_x.cpu().numpy())
            Ys.append(batch_y.cpu().numpy())
            X_ts.append(batch_x_timestamp.cpu().numpy())
            Y_ts.append(batch_y_timestamp.cpu().numpy())
        Xs = np.concatenate(Xs, axis=0)
        Ys = np.concatenate(Ys, axis=0)
        X_ts = np.concatenate(X_ts, axis=0)
        Y_ts = np.concatenate(Y_ts, axis=0)
        print(f"Xs.shape = {Xs.shape} for {flag}_{args.dataset}")
        np.savez(f"dataset/npz/{flag}_{args.dataset}_{args.features}_pl{args.pred_len}_sl{args.seq_len}",
                 Xs=Xs, Ys=Ys, X_ts=X_ts, Y_ts=Y_ts)


if __name__ == "__main__":
    os.makedirs("dataset/npz", exist_ok=True)
    res = []
    args = get_args()
    args.fdir, args.fname = FDIR_DICT[args.dataset]
    get_npz_datasets(args)

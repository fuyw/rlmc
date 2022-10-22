import os
import torch
import numpy as np
from tqdm import tqdm, trange

from models import Autoformer, ETSformer, FEDformer, Informer, Transformer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#######################
# Geriment Settings #
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
    parser.add_argument('--seed', type=int, default=0)

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
    parser.add_argument('--lr', type=float, default=1e-4, help='lr')
    parser.add_argument('--lradj', type=str, default='type1',
                        help='adjust learning rate')

    args = parser.parse_args()
    return args


def get_predictions(args):
    # load scaler
    dataset_name = f"{args.dataset}_{args.features}_pl{args.pred_len}_sl{args.seq_len}"
    ds_scaler = np.load(f"saved_models/scalers/{dataset_name}_0_0.5.npz")
    mu, std = ds_scaler["mu"], ds_scaler["std"]

    # load dataset
    train_dataset = np.load(f"dataset/npz/train_{dataset_name}.npz")
    valid_dataset = np.load(f"dataset/npz/valid_{dataset_name}.npz")
    test_dataset = np.load(f"dataset/npz/test_{dataset_name}.npz")

    # compute batch number
    batch_size = 32
    train_batch_num = int(np.ceil(len(train_dataset["Xs"]) / batch_size))
    valid_batch_num = int(np.ceil(len(valid_dataset["Xs"]) / batch_size))
    test_batch_num  = int(np.ceil(len(test_dataset["Xs"]) / batch_size))
    print(f"[{args.dataset}] Batch num for train/valid/test dataset: "
          f"{train_batch_num}/{valid_batch_num}/{test_batch_num}")

    model_outputs_dict = {}
    for model_name in tqdm(["FEDformer", "Autoformer", "ETSformer", "Informer", "Transformer"]):
        args.model = model_name
        for seed in [0, 1]:
            args.seed = seed
            model_outputs = [[], [], []]        
            args.d_layers = 2 if args.model == "ETSformer" else 1
            model = MODEL_DICT[args.model].Model(args).float()
            ckpt_dir = f"saved_models/{args.dataset}/{args.model}_{args.features}_s{args.seed}_pl{args.pred_len}_sl{args.seq_len}.ckpt"
            model.load_state_dict(torch.load(ckpt_dir))
            model = model.to(device)
            for idx, (total_batch_num, dataset) in enumerate([(train_batch_num, train_dataset),
                                                              (valid_batch_num, valid_dataset),
                                                              (test_batch_num,  test_dataset)]):
                for i in trange(total_batch_num):
                    # tsformer takes (0, 1) scaled inputs
                    batch_x = torch.FloatTensor((dataset["Xs"][i*batch_size:(i+1)*batch_size]-mu)/std).to(device)
                    if "DLinear" == args.model:
                        outputs = model(batch_x)
                    else:
                        batch_y = torch.FloatTensor((dataset["Ys"][i*batch_size:(i+1)*batch_size]-mu)/std).to(device)
                        batch_x_timestamp = torch.FloatTensor(
                            dataset["X_ts"][i*batch_size: (i+1)*batch_size]).to(device)
                        batch_y_timestamp = torch.FloatTensor(
                            dataset["Y_ts"][i*batch_size: (i+1)*batch_size]).to(device)

                        # decoder input
                        zeros_input = torch.zeros_like(batch_y[:, -args.pred_len:, :])
                        decoder_input = torch.cat([batch_y[:, :args.label_len, :],
                                                  zeros_input], dim=1).to(device)

                        outputs = model(batch_x,
                                        batch_x_timestamp,
                                        decoder_input,
                                        batch_y_timestamp)  # (32, 24, 7)
                    model_outputs[idx].append(outputs.detach().cpu().numpy())  # (32, 24, 7)
            train_outputs = np.concatenate(model_outputs[0], axis=0)
            valid_outputs = np.concatenate(model_outputs[1], axis=0)
            test_outputs  = np.concatenate(model_outputs[2], axis=0)

            # inverse transform
            model_outputs_dict[f"{model_name}_s{seed}"] = {
                "train": train_outputs * std + mu,
                "valid": valid_outputs * std + mu,
                "test":  test_outputs * std + mu,
            }

    save_dir = f"dataset/basemodel_predictions/{exp_name}"
    for flag in ["train", "valid", "test"]:
        np.savez(
            f"{save_dir}/{flag}",
            FEDformer_s0=model_outputs_dict["FEDformer_s0"][flag],
            Autoformer_s0=model_outputs_dict["Autoformer_s0"][flag],
            ETSformer_s0=model_outputs_dict["ETSformer_s0"][flag],
            Informer_s0=model_outputs_dict["Informer_s0"][flag],
            Transformer_s0=model_outputs_dict["Transformer_s0"][flag],
            FEDformer_s1=model_outputs_dict["FEDformer_s1"][flag],
            Autoformer_s1=model_outputs_dict["Autoformer_s1"][flag],
            ETSformer_s1=model_outputs_dict["ETSformer_s1"][flag],
            Informer_s1=model_outputs_dict["Informer_s1"][flag],
            Transformer_s1=model_outputs_dict["Transformer_s1"][flag],
        )


if __name__ == "__main__":
    args = get_args()
    exp_name = f"{args.dataset}_{args.features}_pl{args.pred_len}_sl{args.seq_len}"    
    os.makedirs(f"dataset/basemodel_predictions/{exp_name}", exist_ok=True)
    args.fdir, args.fname = FDIR_DICT[args.dataset]
    args.enc_in = args.dec_in = args.c_out = DIMS_DICT[args.dataset]
    get_predictions(args)

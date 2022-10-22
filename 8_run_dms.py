"""If the agent only select one base model, then it is a tabular RL problem.
"""
import os
import copy
import random
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import trange
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm


MODELS = ['FEDformer', 'Autoformer', 'ETSformer', 'Informer', 'Transformer']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIMS_DICT = {"ETTh1": 7,
             "ETTh2": 7,
             "ETTm1": 7,
             "ETTm2": 7,
             "electricity": 321,
             "traffic": 862,
             "weather": 21,
             "exchange_rate": 8}


###################
# Utils Functions #
###################
def load_processed_data(fname="ETTh1_M_pl24_sl96"):
    pred_len = int(fname.split("pl")[1].split("_")[0])
    train_dataset = np.load(f"dataset/npz/train_{fname}.npz")
    valid_dataset = np.load(f"dataset/npz/valid_{fname}.npz")
    test_dataset  = np.load(f"dataset/npz/test_{fname}.npz")
    train_num = len(train_dataset["Xs"])  # (6026)
    valid_num = len(valid_dataset["Xs"])  # (800)
    test_num  = len(test_dataset["Xs"])   # (1719)
    feat_dim = -1 if fname.split("_")[1] == "MS" else 0

    # basemodel prediction outputs
    train_model_outputs = np.load(f"dataset/basemodel_predictions/{fname}/train.npz")
    valid_model_outputs = np.load(f"dataset/basemodel_predictions/{fname}/valid.npz")
    test_model_outputs  = np.load(f"dataset/basemodel_predictions/{fname}/test.npz")

    # input time-series features & targets
    train_X = train_dataset["Xs"]  # (6026, 96, 7)
    valid_X = valid_dataset["Xs"]  # (800, 96, 7)
    test_X  = test_dataset["Xs"]   # (1719, 96, 7)
    train_Y = train_dataset["Ys"][:, -pred_len:, feat_dim:]  # (6026, 24, 7)
    valid_Y = valid_dataset["Ys"][:, -pred_len:, feat_dim:]  # (800, 24, 7)
    test_Y  = test_dataset["Ys"][:, -pred_len:, feat_dim:]   # (1719, 24, 7)

    train_Ts = train_dataset["X_ts"]  # (N, 96, 7)
    valid_Ts = valid_dataset["X_ts"]
    test_Ts  = test_dataset["X_ts"]

    # prediction error
    train_error, valid_error, test_error = [], [], []
    model_train_Ys, model_valid_Ys, model_test_Ys = [], [], []
    for model in MODELS:
        for seed in [0, 1]:
            model_name = f"{model}_s{seed}"

            # single model prediction
            model_train_Y = train_model_outputs[model_name][:, :, feat_dim:]  # (6026, 24, 7)
            model_valid_Y = valid_model_outputs[model_name][:, :, feat_dim:]  # (800, 24, 7)
            model_test_Y  = test_model_outputs[model_name][:, :, feat_dim:]   # (1719, 24, 7)
            model_train_Ys.append(model_train_Y.reshape(train_num, 1, -1))
            model_valid_Ys.append(model_valid_Y.reshape(valid_num, 1, -1))
            model_test_Ys.append(model_test_Y.reshape(test_num, 1, -1))
            # single model prediction error
            model_train_error = np.abs((train_Y - model_train_Y).reshape(train_num, -1)).mean(1)
            model_valid_error = np.abs((valid_Y - model_valid_Y).reshape(valid_num, -1)).mean(1)
            model_test_error  = np.abs((test_Y - model_test_Y).reshape(test_num, -1)).mean(1)
            train_error.append(model_train_error.reshape(-1, 1))
            valid_error.append(model_valid_error.reshape(-1, 1))
            test_error.append(model_test_error.reshape(-1, 1))

    train_error = np.concatenate(train_error, axis=1)  # (6026, 10)
    valid_error = np.concatenate(valid_error, axis=1)  # (800, 10)
    test_error  = np.concatenate(test_error, axis=1)   # (1719, 10)
    model_train_Ys = np.concatenate(model_train_Ys, axis=1)  # (1719, 10, 168)
    model_valid_Ys = np.concatenate(model_valid_Ys, axis=1)  # (1719, 10, 168)
    model_test_Ys = np.concatenate(model_test_Ys, axis=1)  # (1719, 10, 168)
    return (train_X, valid_X, test_X,
            train_Ts, valid_Ts, test_Ts,
            train_error, valid_error, test_error,
            train_Y, valid_Y, test_Y,
            model_train_Ys, model_valid_Ys, model_test_Ys)


def evaluate_agent(agent, test_states, test_bm_preds, test_y):
    weights = []
    batch_num = int(np.ceil(len(test_states) / 512))
    for i in range(batch_num):
        batch_states = test_states[i*512:(i+1)*512]
        with torch.no_grad():
            batch_weights = agent.select_action(batch_states)
            weights.append(batch_weights)
    weights = np.concatenate(weights, axis=0)
    weights = np.expand_dims(weights, axis=-1)  # (2816, 10, 1)
    weighted_y = weights * test_bm_preds   # (2816, 10, 24)
    weighted_y = weighted_y.sum(1)         # (2816, 24)
    test_y = test_y.reshape(len(weighted_y), -1)
    mae_loss = np.abs(test_y - weighted_y).mean()
    mse_loss = np.square(test_y - weighted_y).mean()
    smape_loss = (np.abs(test_y - weighted_y) / (np.abs(test_y) + np.abs(weighted_y)) * 2).mean()
    return mse_loss, mae_loss, smape_loss


#########
# Model #
#########
class TSEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in,
                                                  configs.d_model,
                                                  configs.embed,
                                                  configs.freq,
                                                  configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(mask_flag=False,
                                        factor=configs.factor,
                                        attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention,
                                        configs=configs),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        self.feat_dim = configs.enc_in

    def forward(self, x):
        x_enc, x_mark_enc = x.split(self.feat_dim, dim=-1)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        # enc_out = enc_out.mean(dim=1)
        enc_out = enc_out[:, -1, :]
        return enc_out


class QNetwork(nn.Module):
    def __init__(self, configs, act_dim):
        super().__init__()
        self.encoder = TSEncoder(configs)
        self.out_layer = nn.Linear(configs.d_model, act_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(x)
        x = self.out_layer(x)
        return x


class DQNAgent:
    def __init__(self,
                 configs,
                 act_dim,
                 states,
                 tau=0.005):
        self.act_dim = act_dim
        self.tau = tau
        self.qnet = QNetwork(configs, act_dim).to(device)
        self.target_qnet = copy.deepcopy(self.qnet)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=1e-3)
        self.states = states

    def select_action(self, states):
        actions = np.zeros(shape=(len(states), self.act_dim))
        with torch.no_grad():
            q_values = self.qnet(states.to(device)).cpu().data.numpy()
            action_idx = q_values.argmax(1)
        actions[np.arange(len(states)), action_idx] = 1.
        return actions

    def update(self, batch_state_idxes, batch_actions, batch_rewards):
        batch_next_state_idxes = batch_state_idxes + 1
        batch_states = self.states[batch_state_idxes].to(device)
        batch_next_states = self.states[batch_next_state_idxes].to(device)
        self.optimizer.zero_grad()

        # compute the target Q value
        with torch.no_grad():
            target_q = self.target_qnet(batch_next_states).max(1)[0]
            target_q = batch_rewards + 0.99 * target_q
        current_q = self.qnet(batch_states).gather(-1, batch_actions).squeeze()

        q_loss = F.mse_loss(current_q, target_q)
        q_loss.backward()
        self.optimizer.step()

        # Update the target network
        for param, target_param in zip(
                self.qnet.parameters(), self.target_qnet.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


class Env:
    def __init__(self, train_error, bm_preds, train_y):
        self.error = train_error
        self.bm_preds = bm_preds
        self.y = train_y
        self.eye = np.eye(train_error.shape[1])

    def reward_func(self, idx, action):
        if isinstance(action, int):
            action = self.eye[action]
        weighted_y = np.multiply(action.reshape(-1, 1), self.bm_preds[idx])
        weighted_y = weighted_y.sum(axis=0)

        new_mae = np.abs(weighted_y - self.y[idx].reshape(-1)).mean()
        new_error = np.array([*self.error[idx], new_mae])
        rank = np.where(np.argsort(new_error) == len(new_error) - 1)[0][0]
        return rank, new_mae


##################
# Run Experiment #
##################
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


def run_exp(args, fname="ETTh1_MS_pl24_sl96"): 
    (
        train_X, valid_X, test_X,
        train_Ts, valid_Ts, test_Ts,
        train_error, _, _,
        train_Y, valid_Y, test_Y,
        model_train_Ys, model_valid_Ys, model_test_Ys
    ) = load_processed_data(fname=fname)
    train_input = np.concatenate([train_X, train_Ts], axis=-1)
    valid_input = np.concatenate([valid_X, valid_Ts], axis=-1)
    test_input  = np.concatenate([test_X, test_Ts], axis=-1)
    L = len(train_X) - 1
    states       = torch.FloatTensor(train_input)
    valid_states = torch.FloatTensor(valid_input)
    test_states  = torch.FloatTensor(test_input)

    act_dim = train_error.shape[1]

    env = Env(train_error, model_train_Ys, train_Y)
    if not os.path.exists(f"dataset/buffers/{fname}_buffer.csv"):
        batch_buffer = []
        for state_idx in range(L):
            for action_idx in range(act_dim):
                rank, mae = env.reward_func(state_idx, action_idx)
                batch_buffer.append((state_idx, action_idx, rank, mae))
        batch_buffer_df = pd.DataFrame(
            batch_buffer, columns=["state_idx", "action_idx", "rank", "mae"])
        batch_buffer_df.to_csv(f"dataset/buffers/{fname}_buffer.csv")
    else:
        batch_buffer_df = pd.read_csv(f"dataset/buffers/{fname}_buffer.csv", index_col=0)
    batch_buffer_df["reward"] = -batch_buffer_df["mae"]

    batch_size = 512
    batch_num = int(np.ceil(len(batch_buffer_df) / batch_size))
    best_mae_loss = np.inf
    patience, max_patience = 0, 3
    agent = DQNAgent(args, act_dim, states)
    best_qnet = QNetwork(args, act_dim).to(device)

    for _ in range(15):
        shuffle_idx = np.random.permutation(np.arange(len(batch_buffer_df)))
        for i in range(batch_num):
            batch_idx = shuffle_idx[i*batch_size: (i+1)*batch_size]
            batch_state_idxes = batch_buffer_df.iloc[batch_idx]['state_idx'].values
            batch_actions = torch.LongTensor(batch_buffer_df.iloc[batch_idx]["action_idx"].values).to(device)
            batch_rewards = torch.FloatTensor(batch_buffer_df.iloc[batch_idx]["reward"].values).to(device)
            _ = agent.update(batch_state_idxes, batch_actions.view(-1, 1), batch_rewards)
        _, valid_mae_loss, _ = evaluate_agent(agent, valid_states, model_valid_Ys, valid_Y)
        if valid_mae_loss < best_mae_loss: 
            best_mae_loss = valid_mae_loss
            patience = 0
            best_qnet = copy.deepcopy(agent.qnet)
        else:
            patience += 1
        if patience == max_patience:
            break

    agent.qnet = best_qnet
    test_mse_loss, test_mae_loss, test_smape_loss = evaluate_agent(
        agent, test_states, model_test_Ys, test_Y)
    return test_mse_loss, test_mae_loss, test_smape_loss


def run(args, fname):
    mse_res, mae_res, smape_res = [], [], []
    for i in range(5):
        rseed = np.random.randint(0, 10000)
        np.random.seed(rseed)
        _, mae, smape = run_exp(args, fname)
        mae_res.append(mae)
        smape_res.append(smape)
    res_df = pd.DataFrame({
        "mae": mae_res,
        "smape": smape_res,
    })
    res_df.to_csv(f"results/{args.dataset}/dms_{fname}.csv")

if __name__ == "__main__":
    args = get_args()
    fname = f"{args.dataset}_{args.features}_pl{args.pred_len}_sl96"
    args.enc_in = DIMS_DICT[args.dataset]
    os.makedirs(f"dataset/buffers", exist_ok=True)
    print(f"Run exp: {fname}")
    run(args, fname)

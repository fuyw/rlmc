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
def load_processed_data(fname="ETTh1_MS_pl24_sl96", mask_len=5):
    pred_len = int(fname.split("pl")[1].split("_")[0])
    train_dataset = np.load(f"dataset/npz/train_{fname}.npz")
    valid_dataset = np.load(f"dataset/npz/valid_{fname}.npz")
    test_dataset  = np.load(f"dataset/npz/test_{fname}.npz")
    train_num = len(train_dataset["Xs"])
    valid_num = len(valid_dataset["Xs"])
    test_num  = len(test_dataset["Xs"])
    feat_dim = -1 if fname.split("_")[1] == "MS" else 0
    train_model_outputs = np.load(f"dataset/basemodel_predictions/{fname}/train.npz")
    valid_model_outputs = np.load(f"dataset/basemodel_predictions/{fname}/valid.npz")
    test_model_outputs  = np.load(f"dataset/basemodel_predictions/{fname}/test.npz")
    train_X = train_dataset["Xs"]
    valid_X = valid_dataset["Xs"]
    test_X  = test_dataset["Xs"]
    train_Y = train_dataset["Ys"][:, -pred_len:, feat_dim:]
    valid_Y = valid_dataset["Ys"][:, -pred_len:, feat_dim:]
    test_Y  = test_dataset["Ys"][:, -pred_len:, feat_dim:]
    train_Ts = train_dataset["X_ts"]
    valid_Ts = valid_dataset["X_ts"]
    test_Ts  = test_dataset["X_ts"]
    train_mae_error, train_smape_error = [], []
    model_train_Ys, model_valid_Ys, model_test_Ys = [], [], []
    for model in MODELS:
        for seed in [0, 1]:
            model_name = f"{model}_s{seed}"
            model_train_Y = train_model_outputs[model_name][:, :, feat_dim:]
            model_valid_Y = valid_model_outputs[model_name][:, :, feat_dim:]
            model_test_Y  = test_model_outputs[model_name][:, :, feat_dim:]
            model_train_Ys.append(model_train_Y.reshape(train_num, 1, -1))
            model_valid_Ys.append(model_valid_Y.reshape(valid_num, 1, -1))
            model_test_Ys.append(model_test_Y.reshape(test_num, 1, -1))
            true_ = train_Y.reshape(train_num, -1)
            pred_ = model_train_Y.reshape(train_num, -1)
            model_train_mae_error = np.abs(pred_ - true_).mean(1)
            model_train_smape_error = (np.abs(pred_ - true_) / (np.abs(pred_) + np.abs(true_)) * 2.).mean(1)
            train_mae_error.append(model_train_mae_error.reshape(-1, 1))
            train_smape_error.append(model_train_smape_error.reshape(-1, 1))
    train_mae_error = np.concatenate(train_mae_error, axis=1)
    train_smape_error = np.concatenate(train_smape_error, axis=1)
    model_train_Ys = np.concatenate(model_train_Ys, axis=1)
    model_valid_Ys = np.concatenate(model_valid_Ys, axis=1)
    model_test_Ys = np.concatenate(model_test_Ys, axis=1)
    mask_idx = np.argsort(train_mae_error.mean(0))[:mask_len]
    return (train_X, valid_X, test_X,
            train_Ts, valid_Ts, test_Ts,
            train_mae_error[:, mask_idx], train_smape_error[:, mask_idx],
            train_Y, valid_Y, test_Y,
            model_train_Ys[:, mask_idx, :], model_valid_Ys[:, mask_idx, :], model_test_Ys[:, mask_idx, :])


def load_mu_std(fname):
    ds = np.load(f"saved_models/scalers/{fname}.npz")
    mu, std = ds["mu"], ds["std"]
    return mu, std


def normalize(Xs, mu, std):
    res = []
    for X in Xs:
        res.append((X-mu)/std)
    return res


def evaluate_agent(agent, test_states, test_bm_preds, test_y):
    agent.actor.eval()
    weights = []
    batch_num = int(np.ceil(len(test_states) / 512))
    for i in range(batch_num):
        batch_states = test_states[i*512:(i+1)*512]
        with torch.no_grad():
            batch_weights = agent.select_action(batch_states)
            weights.append(batch_weights)
    weights = np.concatenate(weights, axis=0)
    max_weight = weights.max(1).mean()
    act_counter = Counter(weights.argmax(1))
    act_sorted  = sorted([(k, v) for k, v in act_counter.items()])
    weights = np.expand_dims(weights, axis=-1)
    weighted_y = weights * test_bm_preds
    weighted_y = weighted_y.sum(1)
    test_y = test_y.reshape(len(weighted_y), -1)
    mae_loss = np.abs(test_y - weighted_y).mean()
    mse_loss = np.square(test_y - weighted_y).mean()
    smape_loss = (np.abs(test_y - weighted_y) / (np.abs(test_y) + np.abs(weighted_y)) * 2).mean()
    return mse_loss, mae_loss, smape_loss, max_weight, act_sorted


def sparse_explore(obs, act_dim, alpha=0.1):
    N = len(obs)
    x = np.zeros((N, act_dim))
    randn_idx = np.random.randint(0, act_dim, size=(N,))
    if random.random() < 0.75:
        x[np.arange(N), randn_idx] = 1
        delta = np.random.uniform(0.05, alpha, size=(N, 1))
        x[np.arange(N), randn_idx] -= delta.squeeze()
        noise = np.abs(np.random.randn(N, act_dim))
        noise[np.arange(N), randn_idx] = 0
        noise /= noise.sum(1, keepdims=True)
        noise = delta * noise
        sparse_action = x + noise
    else:
        sparse_action = np.ones(shape=(N, act_dim)) / act_dim
    return sparse_action


def get_loss_quantiles(losses, q=10):
    losses = losses.reshape(-1)
    quantiles = np.array([np.quantile(losses, 0.1*i) for i in range(1, q)])
    return quantiles


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
        enc_out = enc_out[:, -10:, :].mean(dim=1)
        return enc_out


class Actor(nn.Module):
    def __init__(self, configs, act_dim):
        super().__init__()
        self.encoder = TSEncoder(configs)
        self.fc_mu = nn.Linear(configs.d_model, act_dim)
        self.fc_std = nn.Linear(configs.d_model, act_dim)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_std = self.fc_std(x)
        log_std = torch.clip(log_std, -10., 2.)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        return dist


class DoubleCritic(nn.Module):
    def __init__(self, configs, act_dim):
        super().__init__()
        self.encoder1 = TSEncoder(configs)
        self.act_layer1 = nn.Linear(act_dim, configs.d_model)
        self.out_layer1 = nn.Linear(configs.d_model, 1)

        self.encoder2 = TSEncoder(configs)
        self.act_layer2 = nn.Linear(act_dim, configs.d_model)
        self.out_layer2 = nn.Linear(configs.d_model, 1)

    def forward(self, obs, act):
        x1 = self.encoder1(obs) + self.act_layer1(act)
        q1 = self.out_layer1(F.relu(x1))
        x2 = self.encoder2(obs) + self.act_layer2(act)
        q2 = self.out_layer2(F.relu(x2))
        return q1.squeeze(), q2.squeeze()


class RLMCAgent:
    def __init__(self,
                 configs,
                 act_dim,
                 states,
                 tau=0.005,
                 entropy_alpha=0.5):
        self.act_dim = act_dim
        self.tau = tau
        self.actor = Actor(configs, act_dim).to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=configs.lr)
        self.critic = DoubleCritic(configs, act_dim).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=configs.lr)
        self.log_alpha = torch.tensor(np.log(0.01), requires_grad=True)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.states = states
        self.target_entropy = -act_dim * entropy_alpha

    def select_action(self, obs):
        with torch.no_grad():
            dist = self.actor(obs.to(device))
            action = dist.rsample().detach().cpu()
            action = torch.tanh(action).numpy()
        return softmax(action, axis=-1)

    def update_critic(self, batch_state_idxes, batch_actions, batch_rewards):
        batch_next_state_idxes = batch_state_idxes + 1
        batch_states = self.states[batch_state_idxes].to(device)
        batch_next_states = self.states[batch_next_state_idxes].to(device)
        alpha = self.log_alpha.exp().detach()
        next_dist = self.actor(batch_next_states)
        next_action = next_dist.rsample()
        log_prob = next_dist.log_prob(next_action)
        real_next_action = torch.tanh(next_action)
        real_next_log_prob = log_prob - torch.log(1 - torch.tanh(next_action).pow(2) + 1e-7)
        real_next_log_prob = real_next_log_prob.sum(-1)
        next_q1, next_q2 = self.critic(batch_next_states, real_next_action)
        next_q = torch.min(next_q1, next_q2) - alpha * real_next_log_prob
        target_q = batch_rewards + 0.99 * next_q
        q1, q2 = self.critic(batch_states, batch_actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return {"critic_loss": critic_loss.item(),
                "q1": q1.mean().item(),
                "q2": q2.mean().item()}

    def update_actor(self, batch_state_idxes):
        batch_states = self.states[batch_state_idxes].to(device)
        dist = self.actor(batch_states)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        sampled_action = torch.tanh(action)
        logp = log_prob - torch.log(
            1-torch.tanh(sampled_action).pow(2) + 1e-7)
        logp = logp.sum(-1)
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -self.log_alpha * (
            self.target_entropy + logp).detach().mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        self.actor_optimizer.zero_grad()
        alpha = self.log_alpha.exp().detach()
        sampled_q1, sampled_q2 = self.critic(batch_states, sampled_action)
        sampled_q = torch.min(sampled_q1, sampled_q2)
        actor_loss = (alpha * logp - sampled_q).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        return {"actor_loss": actor_loss.item(),
                "alpha_loss": alpha_loss.item(),
                "alpha": alpha.item()}

    def update(self, batch_state_idxes, batch_actions, batch_rewards):
        critic_log = self.update_critic(batch_state_idxes, batch_actions, batch_rewards)
        actor_log = self.update_actor(batch_state_idxes)
        for param, target_param in zip(
                self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
        return {**critic_log, **actor_log}


class Env:
    def __init__(self, train_mae_error, train_smape_error, bm_preds, train_y):
        self.mae_error = train_mae_error
        self.smape_error = train_smape_error
        self.mae_quantile = get_loss_quantiles(train_mae_error, q=10)
        self.smape_quantile = get_loss_quantiles(train_smape_error, q=10)
        self.bm_preds = bm_preds
        self.avg_pred = bm_preds.mean(axis=1)
        self.y = train_y
        self.eye = np.eye(train_mae_error.shape[1])

    def get_mae_rank(self, mae):
        if   mae <= self.mae_quantile[0]: return 1
        elif mae <= self.mae_quantile[1]: return 2
        elif mae <= self.mae_quantile[2]: return 3
        elif mae <= self.mae_quantile[3]: return 4
        elif mae <= self.mae_quantile[4]: return 5
        elif mae <= self.mae_quantile[5]: return 6
        elif mae <= self.mae_quantile[6]: return 7
        elif mae <= self.mae_quantile[7]: return 8
        elif mae <= self.mae_quantile[8]: return 9
        return 10

    def get_smape_rank(self, smape):
        if   smape <= self.smape_quantile[0]: return 1
        elif smape <= self.smape_quantile[1]: return 2
        elif smape <= self.smape_quantile[2]: return 3
        elif smape <= self.smape_quantile[3]: return 4
        elif smape <= self.smape_quantile[4]: return 5
        elif smape <= self.smape_quantile[5]: return 6
        elif smape <= self.smape_quantile[6]: return 7
        elif smape <= self.smape_quantile[7]: return 8
        elif smape <= self.smape_quantile[8]: return 9
        return 10

    def reward_func(self, idx, action):
        if isinstance(action, int): action = self.eye[action]
        weighted_y = np.multiply(action.reshape(-1, 1), self.bm_preds[idx])
        weighted_y = weighted_y.sum(axis=0)
        true_ = self.y[idx].reshape(-1)
        new_mae = np.abs(weighted_y - true_).mean()
        new_smape = (np.abs(weighted_y - true_)/(
            np.abs(weighted_y) + np.abs(true_)) * 2.).mean()
        mae_rank = self.get_mae_rank(new_mae)
        smape_rank = self.get_smape_rank(new_smape)
        mae_reward = (mae_rank - 1)/9
        smape_reward = (smape_rank - 1)/9
        return -(mae_reward + smape_reward) * 2.


#################
# Replay Buffer #
#################
class ReplayBuffer:
    def __init__(self, action_dim, max_size=int(1e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((max_size, 1), dtype=np.int32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, step_size=256):
        ind = np.random.randint(self.size, size=step_size)
        states = self.states[ind]
        actions = torch.FloatTensor(self.actions[ind]).to(device)
        rewards = torch.FloatTensor(self.rewards[ind]).to(device)
        return (states.squeeze(), actions, rewards.squeeze())


##################
# Run Experiment #
##################
def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    # basic config
    parser.add_argument('--model', type=str, default='Autoformer')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='ETTh1')
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--features', type=str, default='MS')

    # data loader setting
    parser.add_argument('--lr', type=float, default=1e-4, help='lr')  # 1e-4, 5e-3
    parser.add_argument('--target', type=str, default='OT',
                        help='target feature in S or MS task')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h',
                        help='[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    # Autoformer config
    parser.add_argument('--wavelet', type=int, default=0,
                        help='whether use wavelet in Autoformer')

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

    # Formers 
    parser.add_argument('--e_layers', type=int, default=2,
                        help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1,
                        help='num of decoder layers')
    parser.add_argument('--enc_in', type=int, default=7,
                        help='encoder input size')
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
    parser.add_argument('--step_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--lradj', type=str, default='type1',
                        help='adjust learning rate')

    args = parser.parse_args()
    return args


def run(args,
        fname="ETTh1_MS_pl24_sl96",
        patience=3,
        epsilon=0.4,
        mask_len=8,
        alpha=0.1,
        pretrain=True,
        pt=2,
        entropy_alpha=0.5,
        threshold_reward=-3.0,
        min_hard_size=2000): 
    (
        train_X, valid_X, test_X,
        train_Ts, valid_Ts, test_Ts,
        train_mae_error, train_smape_error,
        train_Y, valid_Y, test_Y,
        model_train_Ys, model_valid_Ys, model_test_Ys
    ) = load_processed_data(fname=fname, mask_len=mask_len)
    mu, std = load_mu_std(f"{fname}_0_0.5")
    train_X, valid_X, test_X = normalize([train_X, valid_X, test_X], mu, std)
    train_input = np.concatenate([train_X, train_Ts], axis=-1)
    valid_input = np.concatenate([valid_X, valid_Ts], axis=-1)
    test_input  = np.concatenate([test_X, test_Ts], axis=-1)
    L = len(train_X) - 1
    states       = torch.FloatTensor(train_input)
    valid_states = torch.FloatTensor(valid_input)
    test_states  = torch.FloatTensor(test_input)
    act_dim = train_mae_error.shape[1]
    best_mae_loss = np.inf
    patience, max_patience = 0, patience
    agent = RLMCAgent(args, act_dim, states, entropy_alpha=entropy_alpha)
    best_actor = agent.actor
    replay_buffer = ReplayBuffer(act_dim, max_size=int(1e5))
    hard_buffer = ReplayBuffer(act_dim, max_size=int(1e4))
    env = Env(train_mae_error=train_mae_error,
              train_smape_error=train_smape_error,
              bm_preds=model_train_Ys,
              train_y=train_Y)

    if pretrain:
        actor = Actor(args, act_dim).to(device)
        optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
        cls_label = train_mae_error.argmin(axis=1)
        batch_num = int(np.ceil(len(train_input)/256))
        for _ in range(pt):
            randn_idx = np.random.permutation(np.arange(len(train_input)))
            for i in range(batch_num):
                optimizer.zero_grad()
                batch_idx = randn_idx[i*256:(i+1)*256]
                batch_state = states[batch_idx].to(device)
                batch_label = torch.LongTensor(cls_label[batch_idx]).to(device)
                output = actor(batch_state).rsample()
                logits = F.log_softmax(torch.tanh(output), dim=1)
                loss = F.nll_loss(logits, batch_label)
                loss.backward()
                optimizer.step()
        for param, target_param in zip(actor.parameters(), agent.actor.parameters()):    
            target_param.data.copy_(param)

    def get_batch_rewards(env, state_idxes, actions):
        rewards = []
        for i, state_idx in enumerate(state_idxes):
            rew = env.reward_func(state_idx, actions[i])
            rewards.append(rew)
        return rewards

    for _ in range(100):
        shuffle_idxes = np.random.randint(0, L, 300) 
        sampled_states = states[shuffle_idxes]
        sampled_actions = sparse_explore(sampled_states, act_dim, alpha)
        sampled_rewards = get_batch_rewards(env, shuffle_idxes, sampled_actions)
        for i in range(len(sampled_states)):
            replay_buffer.add(shuffle_idxes[i], sampled_actions[i], sampled_rewards[i])
 
    step_size, batch_size = 128, 128
    step_num = int(np.ceil(L / step_size))
    best_mae_loss = np.inf
    for epoch in range(20):
        shuffle_idx = np.random.permutation(np.arange(L))
        for i in range(step_num):
            batch_idx = shuffle_idx[i*step_size: (i+1)*step_size]
            for _ in range(4):
                batch_states = states[batch_idx]
                if np.random.random() < epsilon:
                    batch_actions = sparse_explore(batch_states, act_dim, alpha)
                else:
                    batch_actions = agent.select_action(batch_states)
                batch_rewards = get_batch_rewards(env, batch_idx, batch_actions)
                for j in range(len(batch_idx)):
                    replay_buffer.add(batch_idx[j], batch_actions[j], batch_rewards[j])
                    if batch_rewards[j] <= threshold_reward:
                        hard_buffer.add(batch_idx[j], batch_actions[j], batch_rewards[j])
                batch_idx = (batch_idx + 1) % L
                epsilon = max(0.1, epsilon-0.001)

            sampled_obs_idxes, sampled_actions, sampled_rewards = replay_buffer.sample(batch_size)
            _ = agent.update(sampled_obs_idxes, sampled_actions, sampled_rewards)
            if hard_buffer.size >= min_hard_size:
                sampled_obs_idxes, sampled_actions, sampled_rewards = hard_buffer.sample(batch_size)
                _ = agent.update(sampled_obs_idxes, sampled_actions, sampled_rewards)

        _, valid_mae_loss, _, _, _ = evaluate_agent(agent, valid_states, model_valid_Ys, valid_Y)
        if valid_mae_loss < best_mae_loss: 
            best_mae_loss = valid_mae_loss
            patience = 0
            best_actor = copy.deepcopy(agent.actor)
        elif epoch >= 3:
            patience += 1

        if patience == max_patience:
            break

    agent.actor = best_actor
    test_mse_loss, test_mae_loss, test_smape_loss, max_weight, act_count = evaluate_agent(agent, test_states, model_test_Ys, test_Y)
    return test_mse_loss, test_mae_loss, test_smape_loss


def run_param(args,
              fname,
              patience=3,
              epsilon=0.5,
              mask_len=5,
              alpha=0.1,
              entropy_alpha=0.5,
              pt=3):
    mse_res, mae_res, smape_res = [], [], []
    for i in range(5):
        rseed = np.random.randint(0, 10000)
        np.random.seed(rseed)
        mse, mae, smape = run(args,
                              fname,
                              patience,
                              epsilon,
                              mask_len,
                              alpha,
                              entropy_alpha=entropy_alpha, pt=pt)
        mse_res.append(mse)
        mae_res.append(mae)
        smape_res.append(smape)
    return np.mean(mae_res), np.mean(smape_res)


if __name__ == "__main__":
    args = get_args()
    args.enc_in = DIMS_DICT[args.dataset]
    fname = f"{args.dataset}_{args.features}_pl{args.pred_len}_sl96" 
    print(f"Run exp: {fname}")
    mae, smape = run_param(args,
                           fname,
                           patience=3,
                           epsilon=0.3,
                           mask_len=8,
                           alpha=0.2)
    print(f"mae={mae:.3f}, smape={smape:.3f}\n")

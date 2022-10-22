import os
import copy
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import Counter
from scipy.special import softmax
from utils.metrics import metric


MODELS = ['FEDformer', 'Autoformer', 'ETSformer', 'Informer', 'Transformer']


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
    train_X = train_dataset["Xs"]  # (6026, 48, 7)
    valid_X = valid_dataset["Xs"]  # (800, 48, 7)
    test_X  = test_dataset["Xs"]   # (1719, 48, 7)
    train_Y = train_dataset["Ys"][:, -pred_len:, feat_dim:]  # (6026, 24, 7)
    valid_Y = valid_dataset["Ys"][:, -pred_len:, feat_dim:]  # (800, 24, 7)
    test_Y  = test_dataset["Ys"][:, -pred_len:, feat_dim:]   # (1719, 24, 7)

    # prediction error
    train_error, valid_error, test_error = [], [], []
    model_test_Ys = []
    for model in MODELS:
        for seed in [0, 1]:
            model_name = f"{model}_s{seed}"

            # single model prediction
            model_train_Y = train_model_outputs[model_name][:, :, feat_dim:]  # (6026, 24, 7)
            model_valid_Y = valid_model_outputs[model_name][:, :, feat_dim:]  # (800, 24, 7)
            model_test_Y  = test_model_outputs[model_name][:, :, feat_dim:]   # (1719, 24, 7)
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
    model_test_Ys = np.concatenate(model_test_Ys, axis=1)  # (1719, 10, 168)
    return train_X, valid_X, test_X, train_error, valid_error, test_error, test_Y, model_test_Ys


class FFORMA:
    """FFORMA optimizes a weighted average loss function. Here, we do not
    compute a weighted sum of the model outputs.

    Inputs: (time-series features, single model prediction errors).
    """
    def __init__(self, params, pred_errors):
        self.params = params
        self.pred_errors = pred_errors

    def fforma_objective(self, preds, train_data):
        y = train_data.get_label().astype(int)  # data indices
        preds = np.reshape(preds, self.pred_errors[y, :].shape, order='F')
        weights = softmax(preds, axis=1)
        weighted_loss = (weights * self.pred_errors[y, :]).sum(axis=1).reshape((len(y), 1))
        grad = weights * (self.pred_errors[y, :] - weighted_loss)
        hess = self.pred_errors[y, :] * weights * (1 - weights) - grad * weights
        return grad.flatten('F'), hess.flatten('F')

    def fforma_metric(self, preds, train_data):
        y = train_data.get_label().astype(int)
        preds = np.reshape(preds, self.pred_errors[y, :].shape, order='F')
        weights = softmax(preds, axis=1)
        weighted_loss = (weights * self.pred_errors[y, :]).sum(axis=1)
        fforma_loss = weighted_loss.mean()
        return 'fforma-loss', fforma_loss, False

    # 8.365681  1.685062  0.422099
    def fit_lightgbm(self, feats_train, feats_val):
        indices_train = np.arange(len(feats_train))
        shuffled_indices_train = np.random.permutation(indices_train)
        feats_train = feats_train[shuffled_indices_train]

        # sample indices used to compute prediction error
        indices_val = np.arange(len(feats_val)) + len(feats_train)

        train_set = lgb.Dataset(data=feats_train,
                                label=shuffled_indices_train)
        # train_set = lgb.Dataset(data=feats_train, label=indices_train)
        valid_set = lgb.Dataset(data=feats_val, label=indices_val)
        valid_sets = [train_set, valid_set]

        params = copy.deepcopy(self.params)
        gbm = lgb.train(params=params,
                        train_set=train_set,
                        valid_sets=valid_sets,
                        verbose_eval=-1,
                        num_boost_round=100,
                        fobj=self.fforma_objective,
                        feval=self.fforma_metric)
        return gbm


def run(fname="ETTh1_M_pl24_sl96"):
    (train_X, valid_X, test_X, train_error, valid_error, _, test_Y, model_test_Ys) = load_processed_data(fname=fname)
    train_X = train_X.reshape(len(train_X), -1)  # (N, 48, 7) ==> (N, 336)
    valid_X = valid_X.reshape(len(valid_X), -1)
    test_X  = test_X.reshape(len(test_X), -1)
    test_Y  = test_Y.reshape(len(test_Y), -1)

    train_valid_errors = np.concatenate([train_error, valid_error], axis=0)  # (6730, 10)
    fforma_params = {"eta": 0.58,
                     "seed": 42,
                     "num_leaves": 63,
                     "min_data_in_leaf": 20,
                     "num_class": train_valid_errors.shape[1],
                     "force_col_wise": True,
                     "feature_fraction": 0.8,
                     "verbosity": -1}

    mse_res, mae_res, smape_res = [], [], []
    for i in range(5):
        rseed = np.random.randint(0, 10000)
        np.random.seed(rseed)
        fforma_params["seed"] = rseed
        model = FFORMA(fforma_params, train_valid_errors)
        gbm = model.fit_lightgbm(train_X, valid_X)

        # Predict on test set
        predict_logits = gbm.predict(test_X)
        predict_weights = softmax(predict_logits, axis=1)  # (1719, 6)
        predict_weights = np.expand_dims(predict_weights, axis=2)  # (1719, 6, 1)

        # Weighted prediction
        assert len(model_test_Ys) == len(predict_weights)
        weighted_Ys = (model_test_Ys * predict_weights).sum(axis=1)
        mse, mae, smape = metric(weighted_Ys, test_Y)
        mse_res.append(mse)
        mae_res.append(mae)
        smape_res.append(smape)
    res_df = pd.DataFrame({
        "mae": mae_res,
        "smape": smape_res,
    })
    mu_mae, mu_smape = res_df["mae"].mean(), res_df["smape"].mean()
    std_mae, std_smape = res_df["mae"].std(), res_df["smape"].std()
    print(f"mu: {mu_mae:.3f} ({std_mae:.3f}), smape: {mu_smape:.3f} ({std_smape:.3f})")
    res_df.to_csv(f"results/{args.dataset}/fforma_{fname}.csv") 


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ETTh1')
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--pred_len', type=int, default=24)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    os.makedirs("results", exist_ok=True)
    fname = f"{args.dataset}_{args.features}_pl{args.pred_len}_sl96"
    run(fname)

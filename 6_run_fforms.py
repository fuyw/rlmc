import os
import numpy as np
import pandas as pd
import lightgbm as lgb
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


def run(fname="ETTh1_MS_pl24_sl96"):
    (train_X, valid_X, test_X, train_error, valid_error, _, test_Y, model_test_Ys) = load_processed_data(fname=fname)
    train_X = train_X.reshape(len(train_X), -1)  # (N, 48, 7) ==> (N, 336)
    valid_X = valid_X.reshape(len(valid_X), -1)
    test_X  = test_X.reshape(len(test_X), -1)
    test_Y  = test_Y.reshape(len(test_Y), -1)

    # best model idx
    train_Y = train_error.argmin(axis=1)
    valid_Y = valid_error.argmin(axis=1)

    # lgbm dataset
    train_set = lgb.Dataset(data=train_X, label=train_Y)
    valid_set = lgb.Dataset(data=valid_X, label=valid_Y)
    valid_sets = [train_set, valid_set]

    # lgbm params
    fforms_params = {"boosting_type": "gbdt",
                     "objective": "multiclass",
                     "metric": "multi_logloss",
                     "force_col_wise": True,
                     "num_class": train_error.shape[1],
                     "num_leaves": 31,
                     "learning_rate": 0.01,
                     "feature_fraction": 0.9,
                     "bagging_fraction": 0.8,
                     "bagging_freq": 5,
                     "seed": 42,
                     "verbosity": -1}

    # train the model
    mse_res, mae_res, smape_res = [], [], []
    for i in range(5):
        rseed = np.random.randint(0, 10000)
        np.random.seed(rseed)
        fforms_params["seed"] = rseed
        gbm = lgb.train(fforms_params,
                        train_set=train_set,
                        num_boost_round=1000,
                        valid_sets=valid_sets,
                        early_stopping_rounds=20,
                        verbose_eval=-1)
        pred_Ys = gbm.predict(test_X).argmax(axis=1)  # (1719, 1)
        predict_weights = np.zeros((len(pred_Ys), train_error.shape[1]))
        predict_weights[np.arange(len(pred_Ys)), pred_Ys] = 1.
        predict_weights = np.expand_dims(predict_weights, axis=2)  # (1719, 10, 1)

        # Weighted prediction
        assert len(model_test_Ys) == len(predict_weights)
        weighted_Ys = (model_test_Ys * predict_weights).sum(axis=1)

        # inverse transformation
        mse, mae, smape = metric(weighted_Ys, test_Y)
        mse_res.append(mse)
        mae_res.append(mae)
        smape_res.append(smape)
    res_df = pd.DataFrame({
        "mae": mae_res,
        "smape": smape_res,
    })
    res_df.to_csv(f"results/{args.dataset}/fforms_{fname}.csv")


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ETTh2')
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--pred_len', type=int, default=24)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    os.makedirs("results", exist_ok=True)
    fname = f"{args.dataset}_{args.features}_pl{args.pred_len}_sl96"
    run(fname)

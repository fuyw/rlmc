"""Compute baseline errors:
    - uniform
    - single best
"""
import numpy as np
import pandas as pd
MODELS = ['FEDformer', 'Autoformer', 'ETSformer', 'Informer', 'Transformer']


def get_baseline_metrics(fname="ETTh1_M_pl24_sl96"):
    pred_len = int(fname.split("pl")[1].split("_")[0])
    test_dataset  = np.load(f"dataset/npz/test_{fname}.npz")
    test_num  = len(test_dataset["Xs"])   # (1719)
    feat_dim = -1 if fname.split("_")[1] == "MS" else 0

    # basemodel prediction outputs
    test_model_outputs  = np.load(f"dataset/basemodel_predictions/{fname}/test.npz")
    test_Y = test_dataset["Ys"][:, -pred_len:, feat_dim:]   # (1719, 24, 7)

    # prediction error
    test_mae, test_mse, test_smape, model_test_Ys = [], [], [], []
    for model in MODELS:
        for seed in [0, 1]:
            model_name = f"{model}_s{seed}"
            model_test_Y = test_model_outputs[model_name][:, :, feat_dim:]
            model_test_Ys.append(model_test_Y.reshape(test_num, 1, -1))

            # single model prediction error
            model_test_mae = np.abs((test_Y - model_test_Y).reshape(test_num, -1)).mean(1)
            model_test_mse = np.square(test_Y - model_test_Y).reshape(test_num, -1).mean(1)
            model_test_smape = (np.abs(test_Y - model_test_Y) / (np.abs(test_Y) + np.abs(
                model_test_Y)) * 2).reshape(test_num, -1).mean(1)

            test_mae.append(model_test_mae.reshape(-1, 1))
            test_mse.append(model_test_mse.reshape(-1, 1))
            test_smape.append(model_test_smape.reshape(-1, 1))

    # single model error
    test_mae = np.concatenate(test_mae, axis=1)   # (1719, 10)
    test_mse = np.concatenate(test_mse, axis=1)   # (1719, 10)
    test_smape = np.concatenate(test_smape, axis=1)   # (1719, 10)

    single_best_mae = test_mae.mean(axis=0).min()
    single_best_mse = test_mse.mean(axis=0).min()
    single_best_smape = test_smape.mean(axis=0).min()

    # concat all model predictions
    model_test_Ys = np.concatenate(model_test_Ys, axis=1)  # (1719, 10, 168)
    model_avg_Y = model_test_Ys.mean(axis=1)  # (1719, 168)
    test_Y = test_Y.reshape(test_num, -1)

    # avg prediction loss
    avg_mae = np.abs(model_avg_Y - test_Y).mean()
    avg_mse = np.square(model_avg_Y - test_Y).mean()
    avg_smape = (np.abs(model_avg_Y - test_Y) / (np.abs(model_avg_Y) + np.abs(test_Y)) * 2.).mean()

    return single_best_mae, single_best_mse, single_best_smape, \
        avg_mae, avg_mse, avg_smape


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--pred_len', type=int, default=24)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    res, args = [], get_args()
    for dataset in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity"]:
    # for dataset in ["weather", "traffic", "exchange_rate"]:
        fname = f"{dataset}_{args.features}_pl{args.pred_len}_sl96"
        single_best_mae, _, single_best_smape, \
            avg_mae, _, avg_smape = get_baseline_metrics(fname)
        res.append((
            dataset, single_best_mae,
            single_best_smape, avg_mae, avg_smape))
    res_df = pd.DataFrame(res, columns=[
        "dataset", "single_mae", "single_smape",
        "avg_mae", "avg_smape"])
    res_df.to_csv(f"dataset/baseline_{args.features}_pl{args.pred_len}.csv")

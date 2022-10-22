"""
\multirow{8}{*}{ETTh2}
& \multirow{4}{*}{MAE}
    & 12 & 0.317 & 0.356 & 0.356 (0.000) & 0.338 (0.001) & 0.356 (0.000) & \textbf{0.309 (0.004)} \\
    && 24 & 0.317 & 0.356 & 0.356 (0.000) & 0.338 (0.001) & 0.356 (0.000) & \textbf{0.309 (0.004)} \\
    && 36 & 0.356 & 0.369 & 0.385 (0.000) & 0.343 (0.002) & 0.385 (0.001) & \textbf{0.340 (0.004)} \\
    && 48 & 0.356 & 0.369 & 0.385 (0.000) & 0.343 (0.002) & 0.385 (0.001) & \textbf{0.340 (0.004)} \\
\cmidrule{2-9}
& \multirow{4}{*}{sMAPE}
    & 12 & 0.317 & 0.356 & 0.356 (0.000) & 0.338 (0.001) & 0.356 (0.000) & \textbf{0.309 (0.004)} \\
    && 24 & 0.317 & 0.356 & 0.356 (0.000) & 0.338 (0.001) & 0.356 (0.000) & \textbf{0.309 (0.004)} \\
    && 36 & 0.356 & 0.369 & 0.385 (0.000) & 0.343 (0.002) & 0.385 (0.001) & \textbf{0.340 (0.004)} \\
    && 48 & 0.356 & 0.369 & 0.385 (0.000) & 0.343 (0.002) & 0.385 (0.001) & \textbf{0.340 (0.004)} \\
\midrule
"""
import numpy as np
import pandas as pd


ALGOS = ["fforms", "fforma", "dms", "ddpg", "rlmc"]
METRICS = ["single_mae", "single_smape", "avg_mae", "avg_smape"]


def run(dataset, feats):
    res1 = "\\multirow{8}{*}{"+dataset+"}\n& \multirow{4}{*}{MAE}\n"
    res2 = "\cmidrule{2-10}\n& \multirow{4}{*}{sMAPE}\n"
    for pl in [12, 24, 36, 48]:
        fname = f"{dataset}_{feats}_pl{pl}_sl96"
        baseline_df = pd.read_csv(f"dataset/baseline_{feats}_pl{pl}.csv", index_col=0).set_index(["dataset"])
        single_mae, single_smape, avg_mae, avg_smape = baseline_df.loc[dataset, METRICS]
        mu_mae, std_mae = [avg_mae, single_mae], []
        mu_smape, std_smape = [avg_smape, single_smape], []
        for algo in ALGOS:
            df = pd.read_csv(f"results/{dataset}/{algo}_{fname}.csv", index_col=0)
            mu = df.mean(axis=0)
            std = df.std(axis=0)
            mu_mae.append(mu['mae'])
            std_mae.append(std['mae'])
            mu_smape.append(mu['smape'])
            std_smape.append(std['smape'])
        mae_idx = np.argsort(mu_mae)[0]
        smape_idx = np.argsort(mu_smape)[0]
        prefix = "&" if pl == 12 else "&&"
        if mae_idx == 0:
            res1 += f"    {prefix} {pl} & " + "\\textbf{" +f"{avg_mae:.3f}" + "}" + f" & {single_mae:.3f}"
        elif mae_idx == 1:
            res1 += f"    {prefix} {pl} & {avg_mae:.3f} & " + "\\textbf{" + f"{single_mae:.3f}" + "}"
        else:
            res1 += f"    {prefix} {pl} & {avg_mae:.3f} & {single_mae:.3f}"

        if smape_idx == 0:
            res2 += f"    {prefix} {pl} & " + "\\textbf{" +f"{avg_smape:.3f}" + "}" + f" & {single_smape:.3f}"
        elif smape_idx == 1:
            res2 += f"    {prefix} {pl} & {avg_smape:.3f} & " + "\\textbf{" + f"{single_smape:.3f}" + "}"
        else:
            res2 += f"    {prefix} {pl} & {avg_smape:.3f} & {single_smape:.3f}"

        for i in range(2, 7):
            if mae_idx == i:
                res1 += " & \\textbf{" + f"{mu_mae[i]:.3f} ({std_mae[i-2]:.3f})" + "}"
            else:
                res1 += f" & {mu_mae[i]:.3f} ({std_mae[i-2]:.3f})"
            if smape_idx == i:
                res2 += " & \\textbf{" + f"{mu_smape[i]:.3f} ({std_smape[i-2]:.3f})" + "}"
            else: 
                res2 += f" & {mu_smape[i]:.3f} ({std_smape[i-2]:.3f})"

        if pl < 48:
            res1 += "\\\\\n"
            res2 += "\\\\\n"
    res1 += "\\\\" 
    res2 += "\\\\\n\midrule\n"
    print(res1)
    print(res2)


if __name__ == "__main__":
    feats = "MS"
    for dataset in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity"]:
    # for dataset in ["weather", "traffic", "exchange_rate"]:
        run(dataset, feats)

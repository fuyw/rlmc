import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


trues, preds = [], []
for ds in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
    df = pd.read_csv(f"{ds}_MS_pl24_sl96_case.csv", index_col=0)
    trues.append(df["true"].values)
    preds.append(df["pred"].values)


_, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 3))
for i in range(4):
    ax = axes[i//2][i%2]
    ax.plot(np.arange(24), trues[i], label="RLMC")
    ax.plot(np.arange(24), preds[i], label="Pred")
ax.legend()    
plt.tight_layout()
plt.savefig("case_study.png", dpi=360)

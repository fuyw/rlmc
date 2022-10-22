# RLMC

Run the experiment

```sh
for FEATS in MS M
do
    for PL in 12 24 36 48
    do
        for DATASET in ETTh1 ETTh2 ETTm1 ETTm2
        do
            python 5_get_baseline.py --features=$FEATS --pred_len=$PL
            python 6_run_fforms.py --dataset=$DATASET --features=$FEATS --pred_len=$PL
            python 7_run_fforma.py --dataset=$DATASET --features=$FEATS --pred_len=$PL
            python 8_run_dms.py --dataset=$DATASET --features=$FEATS --pred_len=$PL
            python 9_run_ddpg.py --dataset=$DATASET --features=$FEATS --pred_len=$PL
            python 10_run_rlsac.py --dataset=$DATASET --features=$FEATS --pred_len=$PL
        done
    done
done

```
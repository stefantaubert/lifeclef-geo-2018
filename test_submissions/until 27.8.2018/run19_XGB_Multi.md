# Run 19

## Model
XGBoost Multi Model

## Log
MRR-Score: 0.09629068488944718
Started: 2018-05-24 12:34:04.159312
Finished: 2018-05-24 14:35:43.492669
Duration: 121.66min
Suffix: _p32_r12_o0_t0
Traincolumns: chbio_1, chbio_2, chbio_3, chbio_4, chbio_5, chbio_6, chbio_7, chbio_8, chbio_9, chbio_10, chbio_11, chbio_12, chbio_13, chbio_14, chbio_15, chbio_16, chbio_17, chbio_18, chbio_19, etp, alti, awc_top, bs_top, cec_top, crusting, dgh, dimp, erodi, oc_top, pd_top, text, proxi_eau_fast, clc, latitude, longitude
Seed: 4
Split: 0.1
Modelparams:
- objective: binary:logistic
- max_depth: 2
- learning_rate: 0.1
- seed: 4242
- silent: 1
- eval_metric: logloss
- updater: grow_gpu
- predictor: gpu_predictor
- tree_method: gpu_hist
- num_boost_round: 500
- early_stopping_rounds: 10

## Description
- extract all pixel information from the images to a csv file
- use average pixel value of each channel and round to 12 decimals
- used columns "latitude" and "longitude" from occurrences and merge all information to one csv
- split trainset at 0.1 for validation set
- run XGBoost with all feature columns (35)
- predict each species with one model separately (3336 different models)
- training with logloss and early stopping rounds but used depth 2 instead of 3

## Retrieval Type
Mixed (Textual and Visual)

## Validation Score
0.09629068488944718

## Clef ID
ST_13

## Test Score
0.0352431094163906
# Run 18

## Model
XGBoost Single Model

## Log
MRR-Score: 0.08820145534115442
Started: 2018-05-24 10:51:00.094171
Finished: 2018-05-24 12:26:04.535051
Duration: 95.07min
Suffix: _p32_r12_o0_t0
Traincolumns: chbio_1, chbio_2, chbio_3, chbio_4, chbio_5, chbio_6, chbio_7, chbio_8, chbio_9, chbio_10, chbio_11, chbio_12, chbio_13, chbio_14, chbio_15, chbio_16, chbio_17, chbio_18, chbio_19, etp, alti, awc_top, bs_top, cec_top, crusting, dgh, dimp, erodi, oc_top, pd_top, text, proxi_eau_fast, clc, latitude, longitude
Seed: 4
Split: 0.1
Modelparams:
- updater: grow_gpu
- base_score: 0.5
- booster: gbtree
- objective: multi:softprob
- max_depth: 2
- learning_rate: 0.1
- seed: 4242
- silent: 0
- eval_metric: merror
- num_class: 3336
- num_boost_round: 180
- verbose_eval: 1
- predictor: gpu_predictor
- tree_method: gpu_hist

## Description
- extract all pixel information from the images to a csv file
- use average pixel value of each channel and round to 12 decimals
- used columns "latitude" and "longitude" from occurrences and merge all information to one csv
- split trainset at 0.1 for validation set
- run XGBoost with all feature columns (35) with 180 boosting rounds and depth 2

## Retrieval Type
Mixed (Textual and Visual)

## Clef ID
ST_12

## Test Score
0.0342923174436035
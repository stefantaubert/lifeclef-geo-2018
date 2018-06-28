# Run 23

## Model
XGBoost Multi Model with groups

## global_settings
pixel_count = 32
threshold = 9
min_edge_count = 3
use_mean = 1
min_occurence = 0
round_data_ndigits = 2

## Log
MRR-Score: 0.003588005263500999
Started: 2018-05-26 23:17:42.562115
Finished: 2018-05-27 01:35:45.297026
Duration: 138.05min
Suffix: _p32_r2_o0_t9
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
- use average pixel value of each channel and round to 2 decimals (save as csv)
- used columns "latitude" and "longitude" from occurrences and merge all information to one csv
- calculate species groups with similar species:
	- create mean of all values for each species
	- calculate difference between each species
	- take threshold '9' for building groups
	- build groups only with species with have at least '3' similar species
	- result -> 3301 groups (39 species in groups with size greater than 1)
- predict each group with one model separately (3301 different models)
- each group contained several species which were ordered with regard of their probability in the trainset. After predicting a group on the testset we counted this prediction as predicting all classes of the group with descending probabilities.

## Retrieval Type
Mixed (Textual and Visual)

## Clef ID
ST_17

## Validation Score
0.003588005263500999

## Test Score
0.0326038903745639
# Run 5

## Wrong Testdata
True

## Model
XGBoost Regression Model with Groups

## Log
MRR-Score: 0.07794240081046967
Started: 2018-05-07 16:57:35.359079
Finished: 2018-05-07 19:16:08.020040
Duration: 138.54min
Suffix: _p32_r12_o10_t3
Traincolumns: chbio_1, chbio_2, chbio_3, chbio_4, chbio_5, chbio_6, chbio_7, chbio_8, chbio_9, chbio_10, chbio_11, chbio_12, chbio_13, chbio_14, chbio_15, chbio_16, chbio_17, chbio_18, chbio_19, etp, alti, awc_top, bs_top, cec_top, crusting, dgh, dimp, erodi, oc_top, pd_top, text, proxi_eau_fast, clc, latitude, longitude
Seed: 4
Split: 0.1
Modelparams:
- objective: binary:logistic
- max_depth: 4
- learning_rate: 0.1
- seed: 4242
- silent: 1
- eval_metric: logloss
- updater: grow_gpu
- predictor: gpu_predictor
- tree_method: gpu_hist
- num_boost_round: 300
- early_stopping_rounds: 5
=============================

## Description
- extract all pixel information from the images to a csv file
- use average pixel value of each channel and round to 12 decimals (save as csv)
- used columns "latitude" and "longitude" from occurences and merge all information to one csv
- calculate speciesgroups with similar species:
	- remove species witch occur < 10
	- look at the most common values for each species
	- calculate difference
	- take threshold '3' for building groups -> 2090 groups (366 species in groups with size greater than 1)

## Retrieval Type
Mixed (Textual and Visual)

## Old Clef ID
ST_5

## Test Score
0.00798251253440458

## Validation Score
0.07794240081046967
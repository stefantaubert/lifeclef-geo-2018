# Run 11

## Model
Vector Model

## Log
Started: 2018-05-13 22:55:26.400737
Finished: 2018-05-14 16:42:42.498444
Duration: 1067.27 min
Suffix: _p32_r12_o0
'chbio_1', 'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12',
'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18','chbio_19', 
'etp', 'alti', 'awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top', 'text',
'proxi_eau_fast', 'clc', 'latitude', 'longitude'

## Description
- extract all pixel information from the images to a csv file
- use average pixel value of each channel and round to 12 decimals (save as csv)

I have created a vector of the channels for each test/trainingsrow. Then I calculate the difference between the current testrow vector with every trainrow vector. Then I use the length of the resulting vectors to see which row of the trainset is the most similar to the testrow. The class with the highest probability is then the species of this trainrow. The other classes were obtained from other trainrows which were less similar to the testrow (descending with regard to their similarity).

## Retrieval Type
Visual (war eingentlich auch textual weil long und lat dabei waren)

## Other information
Same method as ST_4 but with predicting the testset instead of the trainset.

## Clef ID
ST_5

## Test Score
0.0271210174024472
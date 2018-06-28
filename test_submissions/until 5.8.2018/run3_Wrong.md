# Run 3

## Wrong Testdata
True

## Description
- extract all pixel information from the images to a csv file
- use average pixel value of each channel and round to 12 decimals
- used columns "latitude" and "longitude" from occurences and merge all information to one csv
- split trainset at 0.1 for validation set
- run XGBoost with all feature columns (35)
- predict each species with one model separetely (3336 different models)
- training with logloss and early stopping rounds

## Retrieval Type
Mixed (Textual and Visual)

## Validation Score
9,48%

## Old Clef ID
ST_3

## Test Score
0.0092327534968753
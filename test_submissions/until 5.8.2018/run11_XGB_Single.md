# Run 11

## Wrong Testdata
False

## Description
- extract all pixel information from the images to a csv file
- use average pixel value of each channel and round to 12 decimals
- used columns "latitude" and "longitude" from occurences and merge all information to one csv
- split trainset at 0.1 for validation set
- run XGBoost with all feature columns (35) with 37 boosting rounds

## Retrieval Type
Mixed (Textual and Visual)

## Clef ID
ST_4

## Test Score
0.00846702271205141
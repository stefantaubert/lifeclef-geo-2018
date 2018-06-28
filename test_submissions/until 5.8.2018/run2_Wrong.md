# Run 2

## Wrong Testdata
True

## Description
- extract all pixel information from the images to a csv file
- use average pixel value of each channel and round to 12 decimals
- used columns "latitude" and "longitude" from occurences and merge all information to one csv
- split trainset at 0.1 for validation set
- run XGBoost with all feature columns (35) with 37 boosting rounds

## Retrieval Type
Mixed (Textual and Visual)

## Old Clef ID
ST_2

## Test Score
0.00807691776590113
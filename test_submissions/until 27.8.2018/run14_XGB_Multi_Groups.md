# Run 14

## Model
XGBoost Multi Model with groups

## Description
- extract all pixel information from the images to a csv file
- use average pixel value of each channel and round to 2 decimals (save as csv)
- used columns "latitude" and "longitude" from occurrences and merge all information to one csv
- calculate species groups with similar species:
	- remove species which occur < 10
	- look at the most common values for each species
	- calculate difference
	- take threshold '3' for building groups -> 2148 groups (266 species in groups with size greater than 1)
- predict each group with one model separately (2148 different models)
- each group contained several species which were ordered with regard of their probability in the trainset. After predicting a group on the testset we counted this prediction as predicting all classes of the group with descending probabilities.

## Retrieval Type
Mixed (Textual and Visual)

## Other information
Same method as ST_5 but with predicting the testset instead of the trainset.

## Clef ID
ST_8

## Test Score
0.022000386535744
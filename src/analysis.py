import pandas as pd
import numpy as np
import data_paths

csv = pd.read_csv(data_paths.occurrences_train, error_bad_lines=False, sep=';', low_memory=False)


print("Count of different species:", len(set(csv["species"].values)))
print("Count of different scientificnames:", len(set(csv["scientificname"].values)))
print("Count of different species ids:", len(set(csv["species_glc_id"].values)))

print("Count of rows: ", len(csv.index))
print("Count of columns: ", len(csv.columns.values))
print("Column names: ", csv.columns.values)

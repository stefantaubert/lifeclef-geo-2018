import pandas as pd
import numpy as np
import data_paths

csv = pd.read_csv(data_paths.occurrences_train, sep=';')

species = set(csv["species_glc_id"].values)

print("Count of different species:", len(set(csv["species"].values)))
print("Count of different scientificnames:", len(set(csv["scientificname"].values)))
print("Count of different species ids:", len(species))
print("Last Species:", list(species)[-1])
# hier sieht man dass die Speziesanzahl gleich der letzten Spezies entspricht, also alle Spezies lückenlos vorkommen
print("All species:", species)

print("Count of rows: ", len(csv.index))
print("Count of columns: ", len(csv.columns.values))
print("Column names: ", csv.columns.values)

import pandas as pd

from geo.data_paths import occurrences_train

def analyse_spec():
    csv = pd.read_csv(occurrences_train, sep=';')

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

if __name__ == '__main__':
    ###(890.0, 2259), (775.0, 2090), (912.0, 2078), (956.0, 1838), (981.0, 1808)
    analyse_spec()
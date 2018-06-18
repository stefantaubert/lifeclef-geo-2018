import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from geo.preprocessing.preprocessing import create_datasets
from geo.models.settings import train_val_split
from geo.data_paths import train

def find_rand():
    create_datasets()
    rand = 0
    x_text = pd.read_csv(train)
    y = list(x_text["species_glc_id"])

    found_rand = False

    while not found_rand:
        
        x_train, x_valid, y_train, y_valid = train_test_split(x_text, y, test_size=train_val_split, random_state=rand)
        x_train = x_train.head(10000)
        y_train = y_train[:10000]

        x_valid = x_valid.head(50)
        y_valid = y_valid[:50]

        is_valid = True

        # Entferne Spezies aus dem Validierungsset, falls diese Spezies nicht im Trainingsset vorkommt
        for index in range(0, len(y_valid)):
            species = y_valid[index]
            if species not in y_train:
                is_valid = False

        if not is_valid:
            print("Not ok:", rand)
            rand += 1  
        else:
            found_rand = True

    print("Ok:", rand)

if __name__ == "__main__":
    find_rand()
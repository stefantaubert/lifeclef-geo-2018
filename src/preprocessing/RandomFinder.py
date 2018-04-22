import module_support_pre
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import data_paths_pre as data_paths
import settings_main as settings
import main_preprocessing

main_preprocessing.create_datasets()
rand = 0
x_text = pd.read_csv(data_paths.train, nrows=10000)
y = list(x_text["species_glc_id"])

found_rand = False

while not found_rand:
    
    x_train, x_valid, y_train, y_valid = train_test_split(x_text, y, test_size=settings.train_val_split, random_state=rand)

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

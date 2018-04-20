import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import data_paths_main as data_paths
import settings_main as settings

rand = 0
x_text = np.load(data_paths.x_text)
y = np.load(data_paths.y_ids)

found_rand = False

while not found_rand:
    
    x_train, x_valid, y_train, y_valid = train_test_split(x_text, y, test_size=settings.train_val_split, random_state=rand)

    is_valid = True

    # Entferne Spezies aus dem Validierungsset, falls diese Spezies nicht im Trainingsset vorkommt
    for index in range(0, len(y_valid)):
        item = y_valid[index]
        if item not in y_train:
            is_valid = False

    if not is_valid:
        print("Not ok:", rand)
        rand += 1  
    else:
        found_rand = True

print("Ok:", rand)

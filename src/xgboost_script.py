import time
import pandas as pd
import numpy as np
import data_paths

all_start = time.time()

csv = pd.read_csv(data_paths.occurrences_train, error_bad_lines=False, sep=';', low_memory=False)

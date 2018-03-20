import pandas as pd
import numpy as np

csv = pd.read_csv("D:/dev/Python/life-clef-geo-2018/occurrences_train.csv", encoding="ISO-8859-1", error_bad_lines=False, sep=';')

print(len(csv.columns.values))

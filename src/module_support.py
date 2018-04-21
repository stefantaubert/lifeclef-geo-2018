import os 
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
dirs = next(os.walk(dir_path))[1]

for d in dirs:
    if not d.startswith("."):
        dir_full_path = dir_path + "/" + d
        if dir_full_path not in sys.path:
            sys.path.append(dir_full_path)
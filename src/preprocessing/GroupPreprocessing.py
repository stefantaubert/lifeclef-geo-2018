from tqdm import tqdm
import os
import pickle
import data_paths_pre as data_paths
import pandas as pd
import GroupExtractor
import TextPreprocessing

def map_groups(original, groups):
        name_group_dict = {}
        for name, group in groups.items():
            for species in group:
                name_group_dict[species] = name
        
        result = list(original)
        for i in tqdm(range(len(result))):
            current_species = result[i]
            ### all species have to be in one group
            assert current_species in name_group_dict.keys()
            result[i] = name_group_dict[current_species]
        
        return result

def map():
    if not os.path.exists(data_paths.train_with_groups):
        GroupPreprocessing()._create()
    else:
        print("Groups are already mapped.")

class GroupPreprocessing():
    def __init__(self):
        csv, _, _ = TextPreprocessing.load_train()
        self.csv = csv
        self.named_groups = GroupExtractor.load()
        
    def _create(self):
        print("Mapping species_glc_ids to groups...")
        mapped_species = map_groups(self.csv["species_glc_id"], self.named_groups)
        self.csv["species_glc_id"] = mapped_species
        self.csv.to_csv(data_paths.train_with_groups, index=False)
        
            
if __name__ == "__main__":
    map()
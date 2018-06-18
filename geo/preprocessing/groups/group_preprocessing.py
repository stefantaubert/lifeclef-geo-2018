import os

from geo.data_paths import train_with_groups
from geo.preprocessing.groups.group_extraction import load_named_groups
from geo.preprocessing.groups.group_mapping import map_groups
from geo.preprocessing.text_preprocessing import load_train

def extract_train_with_groups():
    if not os.path.exists(train_with_groups):
        _create()
    else:
        print("Groups are already mapped.")

def _create():
    csv, _, _ = load_train()
    named_groups = load_named_groups()
    print("Mapping species_glc_ids to groups...")
    mapped_species = map_groups(csv["species_glc_id"], named_groups)
    csv["species_glc_id"] = mapped_species
    csv.to_csv(train_with_groups, index=False)

if __name__ == "__main__":
    extract_train_with_groups()
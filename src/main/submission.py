import numpy as np
import data_paths_main as data_paths
import pandas as pd
import submission_maker

def make_xgb_submission():
    print("Make submission...")

    classes = np.load(data_paths.xgb_species_map)

    predictions = np.load(data_paths.xgb_prediction)
    glc_ids = np.load(data_paths.xgb_glc_ids)
    df = submission_maker.make_submission_df(classes, predictions, glc_ids)

    print("Save submission...")
    df.to_csv(data_paths.xgb_submission, index=False)


def make_xgb_groups_submission():
    print("Make submission...")

    groups = np.load(data_paths.xgb_group_map)

    predictions = np.load(data_paths.xgb_prediction)
    glc_ids = np.load(data_paths.xgb_glc_ids)
    species_occ = pd.read_csv(data_paths.xgb_species_occurences)
    named_groups = np.load(data_paths.named_groups)
    
    species_occ_dict = {}
    for _, row in species_occ.iterrows():
        species_occ_dict[row["species"]] = row["percents"]

    submission_df = submission_maker.make_submission_groups_df(groups, predictions, glc_ids, named_groups, species_occ_dict)
    
    print("Save submission...")
    submission_df.to_csv(data_paths.xgb_submission, index=False)

def make_submission_from_files(species_map_path, predictions_path, glc_ids_path, submission_path):
    print("Make submission...")

    classes = np.load(species_map_path)

    predictions = np.load(predictions_path)
    glc_ids = np.load(glc_ids_path)
    df = submission_maker.make_submission_df(classes, predictions, glc_ids)

    print("Save submission...")
    df.to_csv(submission_path, index=False)


if __name__ == '__main__':
    #make_submission_for_current_training()
    make_xgb_groups_submission()
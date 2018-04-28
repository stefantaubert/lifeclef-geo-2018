import numpy as np
import data_paths_main as data_paths
import settings_main as settings
import pandas as pd
import submission_maker
import data_paths_analysis
import SpeciesOccurences

def make_xgb_submission():
    print("Make validation submission...")

    classes = np.load(data_paths.xgb_species_map)

    predictions = np.load(data_paths.xgb_prediction)
    glc_ids = np.load(data_paths.xgb_glc_ids)
    df = submission_maker.make_submission_df(settings.TOP_N_SUBMISSION_RANKS, classes, predictions, glc_ids)

    print("Save validation submission...")
    df.to_csv(data_paths.xgb_submission, index=False)

def make_xgb_test_submission():
    print("Make test submission...")
    classes = np.load(data_paths.xgb_species_map)
    predictions = np.load(data_paths.xgb_test_prediction)
    glc_ids = np.load(data_paths.xgb_test_glc_ids)
    df = submission_maker.make_submission_df(settings.TOP_N_SUBMISSION_RANKS, classes, predictions, glc_ids)

    print("Save test submission...")
    df.to_csv(data_paths.xgb_test_submission, index=False, sep=";")

def make_xgb_groups_submission():
    print("Make validation submission...")

    groups = np.load(data_paths.xgb_group_map)

    predictions = np.load(data_paths.xgb_prediction)
    glc_ids = np.load(data_paths.xgb_glc_ids)
    SpeciesOccurences.create()
    species_occ = pd.read_csv(data_paths_analysis.species_occurences)
    named_groups = np.load(data_paths.named_groups)
    
    species_occ_dict = {}
    for _, row in species_occ.iterrows():
        species_occ_dict[row["species"]] = row["percents"]

    submission_df = submission_maker.make_submission_groups_df(settings.TOP_N_SUBMISSION_RANKS, groups, predictions, glc_ids, named_groups, species_occ_dict)
    
    print("Save validation submission...")
    submission_df.to_csv(data_paths.xgb_submission, index=False)

def make_xgb_groups_test_submission():
    print("Make test submission...")

    groups = np.load(data_paths.xgb_group_map)
    groups = [int(g) for g in groups]

    predictions = np.load(data_paths.xgb_test_prediction)
    glc_ids = np.load(data_paths.xgb_test_glc_ids)

    SpeciesOccurences.create()
    species_occ = pd.read_csv(data_paths_analysis.species_occurences)
    named_groups = np.load(data_paths.named_groups)
    
    species_occ_dict = {}
    for _, row in species_occ.iterrows():
        species_occ_dict[row["species"]] = row["percents"]

    submission_df = submission_maker.make_submission_groups_df(settings.TOP_N_SUBMISSION_RANKS, groups, predictions, glc_ids, named_groups, species_occ_dict)
    
    print("Save test submission...")
    submission_df.to_csv(data_paths.xgb_test_submission, index=False, sep=";")

def make_submission_from_files(species_map_path, predictions_path, glc_ids_path, submission_path):
    print("Make submission...")

    classes = np.load(species_map_path)

    predictions = np.load(predictions_path)
    glc_ids = np.load(glc_ids_path)
    df = submission_maker.make_submission_df(settings.TOP_N_SUBMISSION_RANKS, classes, predictions, glc_ids)

    print("Save submission...")
    df.to_csv(submission_path, index=False, sep=";")
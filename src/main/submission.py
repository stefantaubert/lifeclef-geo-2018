import module_support_main
import numpy as np
import settings_main as settings
import submission_maker

def make_submission_from_files(species_map_path, predictions_path, glc_ids_path, submission_path, header=True):
    print("Make submission...")

    classes = np.load(species_map_path)

    predictions = np.load(predictions_path)
    glc_ids = np.load(glc_ids_path)
    df = submission_maker.make_submission_df(settings.TOP_N_SUBMISSION_RANKS, classes, predictions, glc_ids)

    print("Save submission...")
    if header:
        df.to_csv(submission_path, index=False, sep=";")
    else:  
        df.to_csv(submission_path, index=False, sep=";", header=None)

if __name__ == "__main__":
    make_xgb_submission()
    df.to_csv(submission_path, index=False, sep=";")

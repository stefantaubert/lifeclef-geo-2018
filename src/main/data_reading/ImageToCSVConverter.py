import numpy as np
import pandas as pd
import data_paths
from tqdm import tqdm
import tifffile
import pickle
import settings
import sys
from data_reading.PixelValueExtractor import get_pixel_value
  
class ImageToCSVConverter:
    def __init__(self, pixel_count):
        self.count_channels = 33
        self.image_dimension = 64     
        self.pixel_count = pixel_count  
        self.result_cols = [
            'chbio_1', 'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
            'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12',
            'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18','chbio_19',
            'etp', 'alti','awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top', 'text',
            'proxi_eau_fast', 'clc', 
            'day', 'month', 'year', 'latitude', 'longitude', 'patch_id']

    def _create_dataset_df(self, path_occurences, batch_dir):
        x_text = []
        df = pd.read_csv(path_occurences, sep=';', low_memory=False)
        species_id_column_name = 'species_glc_id'
        is_train = species_id_column_name in df.columns.values
        columns = list(self.result_cols)

        if is_train:
            columns.append(species_id_column_name)
        
        for _, row in tqdm(df.iterrows(), miniters=100):
            current_patch_dirname = row["patch_dirname"]
            current_patch_id = row["patch_id"]

            img = tifffile.imread(batch_dir+'/{}/patch_{}.tif'.format(current_patch_dirname, current_patch_id))
            assert img.shape == (self.count_channels, self.image_dimension, self.image_dimension)

            csv_values = []

            for i in range(len(img)):
                mean = get_pixel_value(img[i], self.pixel_count)
                csv_values.append(mean)

            csv_values.append(row.day)
            csv_values.append(row.month)
            csv_values.append(row.year)
            csv_values.append(row.Latitude)
            csv_values.append(row.Longitude)

            # patch_id != index, da zwischendrin immer mal zeilen mit höheren patch_ids vorkommen
            csv_values.append(row.patch_id)
          
            if is_train:
                csv_values.append(row.species_glc_id)

            x_text.append(csv_values)

        results_array = np.asarray(x_text) #list to array to add to the dataframe as a new column
        result_ser = pd.DataFrame(results_array, columns=columns)
        
        return result_ser

    def _create_train(self):
        df = self._create_dataset_df(data_paths.occurrences_train, data_paths.patch_train)
        df.to_csv(data_paths.occurrences_train_gen, index=False)
   
    def _create_test(self):
        df = self._create_dataset_df(data_paths.occurrences_test, data_paths.patch_train)
        df.to_csv(data_paths.occurrences_test_gen, index=False)

    def create_datasets(self):    
        print("Creating trainset...")
        self._create_train()
        print("Creating testset...")
        self._create_test()

import numpy as np
import pandas as pd
import data_paths
from tqdm import tqdm
import tifffile
import pickle
import settings
import sys

class Data:
    def __init__(self):
        self.count_channels = 33
        self.image_dimension = 64       
        self.result_cols = [
            'chbio_1', 'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6',
            'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12',
            'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18','chbio_19',
            'etp', 'alti','awc_top', 'bs_top', 'cec_top', 'crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top', 'text',
            'proxi_eau_fast', 'clc', 
            'day', 'month', 'year', 'latitude', 'longitude', 'patch_id']

    def get_image_value(self, img, c_pixel):
        '''c_pixel: 1 -> 4 mittlersten Pixel; 2-> 16 innersten pixel; 3 -> 36 innersten pixel
            Berechnung: c_pixel^2 * 4 oder (2 * c_pixel) ^ 2        
            bei 1 -> 4 = (64-2*1) / 2 = 31 for i, j = 2*1
            bei 2 -> 16 = (64-2*2) / 2 = 30 for i, j = 2*2
            bei 3 -> 36 = (64-2*3) / 2 = 29 for i, j = 2*3
        '''
        img_dim = len(img)

        assert c_pixel > 0
        assert c_pixel <= img_dim / 2
        assert img_dim % 2 == 0
        assert img.shape == (img_dim, img_dim)

        if c_pixel == img_dim / 2:
            img_mean = img.mean()
            return img_mean

        dimension = c_pixel * 2
        values = []
        base_index = int((img_dim - (2*c_pixel)) / 2)

        for i in range(dimension):
            for j in range(dimension):
                values.append(img[base_index + i][base_index + j])

        assert len(values) == c_pixel * c_pixel * 4

        values_array = np.asarray(values)
        mean = values_array.mean()

        return mean
        
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

            #Change datatype of img from uint8 to enable mean calculation
            img = np.array(img, dtype=np.uint16)

            csv_values = []

            for i in range(len(img)):
                mean = self.get_image_value(img[i], settings.pixel_count)
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

    def _load_train(self):
        self.train = pd.read_csv(data_paths.occurrences_train_gen)
        self.species = sorted(self.train.species_glc_id.unique())
        self.species_count = len(self.species)
        r = [i + 1 for i in range(len(self.species))]
        assert self.species == r
    
    def _load_test(self):
        self.test = pd.read_csv(data_paths.occurrences_test_gen)

    def create_datasets(self):    
        print("Creating trainset...")
        self._create_train()
        print("Creating testset...")
        self._create_test()

    def load_datasets(self):
        print("Loading trainset...")
        self._load_train()
        print("Loading testset...")
        self._load_test()

if __name__ == "__main__":
    Data().create_datasets()
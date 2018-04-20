import MostCommonValueExtractor
import GroupExtractor
import SimilarSpeciesExtractor
import SpeciesDiffExtractor
import TextPreprocessing
import ImageToCSVConverter
import GroupPreprocessing
import sys
sys.path.append('../')

def extract_groups():
    ImageToCSVConverter.extract_occurences_train()
    TextPreprocessing.extract_train()
    MostCommonValueExtractor.extract()
    SpeciesDiffExtractor.extract()
    SimilarSpeciesExtractor.extract()
    GroupExtractor.extract()
    GroupPreprocessing.map()

def create_datasets():
    ImageToCSVConverter.extract_occurences_train()
    TextPreprocessing.extract_train()
    ImageToCSVConverter.extract_occurences_test()
    TextPreprocessing.extract_test()

if __name__ == "__main__":
    create_datasets()
    #extract_groups()
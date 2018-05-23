import module_support_pre
import MostCommonValueExtractor
import GroupExtractor
import SimilarSpeciesExtractor
import SpeciesDiffExtractor
import TextPreprocessing
import ImageToCSVConverter
import GroupPreprocessing

def extract_groups():
    ImageToCSVConverter.extract_occurences_train()
    TextPreprocessing.extract_train()
    MostCommonValueExtractor.extract()
    SpeciesDiffExtractor.extract()
    SimilarSpeciesExtractor.extract()
    GroupExtractor.extract()
    GroupPreprocessing.map()

def create_trainset():
    ImageToCSVConverter.extract_occurences_train()
    TextPreprocessing.extract_train()

def create_testset():
    ImageToCSVConverter.extract_occurences_test()
    TextPreprocessing.extract_test()

def create_datasets():
    create_trainset()
    create_testset()

if __name__ == "__main__":
    #create_datasets()
    extract_groups()
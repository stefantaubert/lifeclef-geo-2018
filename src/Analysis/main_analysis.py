import module_support_analysis

import GroupAnalysis
import ImageAnalysis
import MostCommonValueDiagram
import MostCommonValuesPerSpeciesDiagram
import SpeciesOccurencesPerValueDiagram
import SpeciesOccurencesPerValueDiagram
import SpeciesOccurences
import SpeciesOccurencesPerValueDiagram
import ValuesOccurencesDiagram
import ValuesOccurencesPerSpeciesDiagram

def run_after_group_changes():
    GroupAnalysis.run()
    
if __name__ == "__main__":
    run_after_group_changes()
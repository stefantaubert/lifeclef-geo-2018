# Hinweis: Einstellungen können nur hier zur Design-Zeit gesetzt werden! Laufzeitänderungen werden ignoriert.

### Gibt die Anzahl an Pixeln^2 pro Quadrant eines Bildes an.
### Minimum: 1 -> entspricht 4 mittlersten Pixel, Maximum: 32 entspricht allen Pixel
pixel_count = 32

### Contains the threshold for the distances between each species for them to be in a group.
### Remarks: after changing this value, you have to run SimilarSpeciesExtractor and GroupExtractor again.
threshold = 20

### Contains the minimal occurence for a species that a species is left in the trainset.
### Remarks: after changing this value, you have to run MostCommonValueExtractor, SpeciesDiffExtractor, SimilarSpeciesExtractor and GroupExtractor again.

min_occurence = 20

### Sets the count of digits for rounding the test and trainset in the analysis. Range: [0, 12]
round_data_ndigits = 0
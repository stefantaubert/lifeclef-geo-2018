# Hinweis: Einstellungen können nur hier zur Design-Zeit gesetzt werden! Laufzeitänderungen werden ignoriert.

BATCH_SIZE = 256

EPOCHS = 10

### Contains the number of top ranks which will be included in the submissionfile.
TOP_N_SUBMISSION_RANKS = 100

### Gibt den Seed für den Split und das Training an.
### Contains the seed which is used to split the trainset and for training
seed = 4 #27 #4 hier ist bei 0.1 jede Spezies aus dem Validierungsset im Trainingsset vorhanden

### Gibt das Split-Verhältnis von Trainings- und Validierungsset an.
### Contains the split ratio for trainset
train_val_split = 0.1


#multi model channels

model1_channel = 0
model2_channel = 12
model3_channel = 13
model4_channel = 4
model5_channel = 20
model6_channel = 21

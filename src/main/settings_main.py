# Hinweis: Einstellungen können nur hier zur Design-Zeit gesetzt werden! Laufzeitänderungen werden ignoriert.
BATCH_SIZE = 128

EPOCHS = 1

### Contains the number of top ranks which will be included in the submissionfile.
TOP_N_SUBMISSION_RANKS = 100

### Gibt den Seed für den Split und das Training an.
### Contains the seed which is used to split the trainset and for training
seed = 4 #27 #4 hier ist bei 0.1 jede Spezies aus dem Validierungsset im Trainingsset vorhanden

### Gibt das Split-Verhältnis von Trainings- und Validierungsset an.
### Contains the split ratio for trainset
train_val_split = 0.1


CHANNEL_COUNT = 33

#multi model channels

MULTI_MODEL_CHANNELS = [0, 12, 13, 4, 20, 21]

model1_channel = 0
model2_channel = 12
model3_channel = 13
model4_channel = 4
model5_channel = 20
model6_channel = 21

resize = False
resize_h = 224
resize_w = 224


#model settings
L2_RATE = 0.1
ACTIVATION = 'relu'


KERAS_MODEL = 'VGG_like'    #Use Vgg like model for KerasModel 
#KERAS_MODEL = 'DenseNet'   #Use DenseNet model for KerasModel

AUGMENT = False
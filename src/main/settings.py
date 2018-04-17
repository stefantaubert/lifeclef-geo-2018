# Hinweis: Einstellungen können nur hier zur Design-Zeit gesetzt werden! Laufzeitänderungen werden ignoriert.

### Gibt den Seed für den Split und das Training an.
seed = 39609 # 4 # hier ist bei 0.1 jede Spezies aus dem Validierungsset im Trainingsset vorhanden

### Gibt das Split-Verhältnis von Trainings- und Validierungsset an.
train_val_split = 0.1

### Gibt die Anzahl an Pixeln^2 pro Quadrant eines Bildes an.
### Minimum: 1 -> entspricht 4 mittlersten Pixel, Maximum: 32 entspricht allen Pixel
pixel_count = 32

BATCH_SIZE = 32

EPOCHS = 30
# Species Prediction based on Environmental Variables using Machine Learning Techniques

By [Stefan Taubert](https://stefantaubert.com/), [Max Mauermann](https://www.tu-chemnitz.de/informatik/HomePages/Medieninformatik/team.php.en), [Stefan Kahl](http://medien.informatik.tu-chemnitz.de/skahl/about/), [Thomas Wilhelm-Stein](https://www.tu-chemnitz.de/informatik/HomePages/Medieninformatik/team.php.en), [Danny Kowerko](https://www.tu-chemnitz.de/informatik/mc/staff.php.en) and [Maximilian Eibl](https://www.tu-chemnitz.de/informatik/HomePages/Medieninformatik/team.php.en)


## Introduction
This is the sourcecode for our submissions to the GeoLifeCLEF 2018 species recognition task.

Contact:  [Stefan Taubert](https://stefantaubert.com/), [Technische Universität Chemnitz](https://www.tu-chemnitz.de/index.html.en), [Media Informatics](https://www.tu-chemnitz.de/informatik/Medieninformatik/index.php.en)

E-Mail: stefan.taubert@informatik.tu-chemnitz.de

This project is licensed under the terms of the MIT license.

Please cite the paper in your publications if it helps your research.
```
@article{taubert2018large,
  title={Species Prediction based on Environmental Variables using Machine Learning Techniques},
  author={Taubert, Stefan and Mauermann, Max and Kahl, Stefan and Wilhelm-Stein, Thomas and Kowerko, Danny and Ritter, Marc and Eibl, Maximilian},
  journal={Working notes of CLEF},
  year={2018}
}
```
<b>You can download our working notes here:</b> [TUCMI GeoLifeCLEF Working Notes PDF](todo)

## Installation
![Python](https://img.shields.io/badge/python-3.6.0-green.svg)

```
git clone git@github.com:stefantaubert/lifeclef-geo-2018.git
cd lifeclef-geo-2018
sudo pip install –r requirements.txt
```

## Training
For the training you need to download the GeoLifeCLEF [training data](http://otmedia.lirmm.fr/LifeCLEF/GeoLifeCLEF2018/).

### Dataset
You need to set up the path to the directory with the datasets. 
Therefor you need to create a file ```geo/data_dir_config.py``` which defines a ```root```-variable and looks like this:
```
root = "/path/to/datasetdir"
```

In this dataset directory should be the following files and directories:
```
occurrences_test.csv
occurrences_train.csv
patchTrain   
¦   256
¦   ¦   patch_1.tif
¦   ¦   patch_2.tif
¦   ¦   ...
¦   512
¦   ¦   patch_257.tif
¦   ¦   patch_258.tif
¦   ¦   ...
¦   ...
patchTest
¦   256
¦   ¦   patch_1.tif
¦   ¦   patch_2.tif
¦   ¦   ...
¦   512
¦   ¦   patch_257.tif
¦   ¦   patch_258.tif
¦   ¦   ...
¦   ...
```

## Run Models
To run any of the eight models you need to navigate to the specific model directory and execute the according python script:

### XGB Single Model
```
PYTHONPATH=/path/to/gitrepo python geo/models/xgb/single_model.py
```

### XGB Multi Model
```
PYTHONPATH=/path/to/gitrepo python geo/models/xgb/multi_model.py
```

### XGB Multi Model with Groups
```
PYTHONPATH=/path/to/gitrepo python geo/models/xgb/multi_model_with_groups.py
```

### Keras Single Model
```
PYTHONPATH=/path/to/gitrepo python geo/models/keras/train_keras_model.py
```

### Keras Multi Model
```
PYTHONPATH=/path/to/gitrepo python geo/models/keras/train_keras_model.py
```

### Vector Model
```
PYTHONPATH=/path/to/gitrepo python geo/models/vector/model.py
```

### Random Model
```
PYTHONPATH=/path/to/gitrepo python geo/models/random/model.py
```

### Probability Model
```
PYTHONPATH=/path/to/gitrepo python geo/models/probability/model.py
```

## Tests
If you want to run the tests you need to run the specific script in the test dir:
```
PYTHONPATH=/path/to/gitrepo python geo/tests/test_*.py
```

## Analysis
You can run our analysis with any script in the ```geo/analysis/``` directory for instance:
```
PYTHONPATH=/path/to/gitrepo python geo/analysis/species_occurences.py
```

## On Windows
Look at this [post](https://stackoverflow.com/a/4580120/3684580) on StackOverflow to set the PYTHONPATH. An other possibility is to use Visual Studio Code and set the ```launch.json``` like this:
```
{
"version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "env": {"PYTHONPATH":"${workspaceRoot}"}
        }
    ]
}
```
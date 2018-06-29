# Species Prediction based on Environmental Variables using Machine Learning Techniques
By [Stefan Taubert](https://stefantaubert.com/), [Max Mauermann](https://www.tu-chemnitz.de/informatik/HomePages/Medieninformatik/team.php.en), [Stefan Kahl](http://medien.informatik.tu-chemnitz.de/skahl/about/), [Thomas Wilhelm-Stein](https://www.tu-chemnitz.de/informatik/HomePages/Medieninformatik/team.php.en), [Danny Kowerko](https://www.tu-chemnitz.de/informatik/mc/staff.php.en) and [Maximilian Eibl](https://www.tu-chemnitz.de/informatik/HomePages/Medieninformatik/team.php.en)


## Introduction
This is the sourcecode for our submissions to the GeoLifeCLEF 2018 species recognition task.

Contact:  [Stefan Taubert](https://stefantaubert.com/), [Technische Universität Chemnitz](https://www.tu-chemnitz.de/index.html.en), [Media Informatics](https://www.tu-chemnitz.de/informatik/Medieninformatik/index.php.en)

E-Mail: stefan.taubert@informatik.tu-chemnitz.de

This project is licensed under the terms of the MIT license.

Please cite the paper in your publications if it helps your research.

```
@article{todo,
  title={Species Prediction based on Environmental Variables using Machine Learning Techniques},
  author={Taubert, Stefan and Mauermann Max and Kahl, Stefan and Wilhelm-Stein, Thomas and Kowerko, Danny and Ritter, Marc and Eibl, Maximilian},
  journal={Working notes of CLEF},
  year={2018}
}
```
<b>You can download our working notes here:</b> [TUCMI GeoLifeCLEF Working Notes PDF](todo)

## Installation

```
git clone git@github.com:stefantaubert/lifeclef-geo-2018.git
cd lifeclef-geo-2018
sudo pip install –r requirements.txt
```

## Training
For the training you need to download the GeoLifeCLEF [training data](http://otmedia.lirmm.fr/LifeCLEF/GeoLifeCLEF2018/).

### Dataset
You need to set the path to the directory with the data. 
Therefor you need to create a file ./data_dir_config.py which defines a root variable and looks like this:

```
root = "path/to/datasetdir"
```

## Run Models
To run the models you need to navigate to the specific model directory and execute the python script for example:

```
cd geo/models/xgb
PYTHONPATH=/path/to/gitrepo python single_model.py
```


# Drift Detection


### Install Dependencies

To create a virtual environment and install the required dependencies, run the following:
``` 
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```
---

### Streaming Simulation on Office 31 Dataset 

To run a streaming simulation on the Office 31 dataset (without online training), invoke the following:

`python3 office_simulator.py <path to reference samples> <path to first set of target samples> <path to second set of target samples> <trained model> <test set filenames for reference distribution>`

Results are written to JSON files in the `out` directory and analysis is provided in `results_office31`.

---

### BDD100K — Drift Detection on Weather Domains

To train a Faster-RCNN model on the `clear` weather domain, run:
```
export TRAIN_PATH=<path to training directory>
export TRAIN_JSON=<path to labels/det_train.json>
python3 train-bdd100k-clear-domain.py
```
---

### BDD100K — Drift Detection on Mapillary, IDD, Cityscapes, and Foggy Cityscapes

A Faster-RCNN model with a `resnet50` backbone was trained on the following object categories:
```
pedestrian
rider
car
truck
bus
train
motorcycle
bicycle
traffic light
traffic sign
```
Download the model weights here and place the file in the `models` directory: https://drive.google.com/file/d/1yJZ0v_dXtUoTeHrJosRXsw-GxwhEpJhC/view?usp=sharing 

Evaluate on the BDD100K validation set:

```
export VAL_PATH=<path to validation set>
export VAL_JSON=<path to labels/det_val.json>
python3 eval-bdd100k.py
```

Evaluate on the Mapillary validation set:
```
export VAL_PATH=<path to validation set>
export VAL_JSON=<path to v2.0/polygons>
python3 eval-mapillary.py
```

To evaluate the model on the Indian Driving Dataset, first create the validation set:
```
export IDD_val_path=<path to val.txt>
export IDD_img_path=<path to dataset directory>
export IDD_annotations_path=<path to annotations directory>
python3 make idd-eval.py
```
Then, run the evaluation script:
```
export IMAGE_PATH=<path to eval-img>
python3 eval-idd.py
```

Evaluate on Foggy Cityscapes (with attenuation coefficient of 0.02):
```
export FINE_ANN=<path to fine annotations>
export IMG_DIR=<path to leftImg8bit_transmittance>
python3 eval-foggy-cityscapes.py
```

Evaluate on Cityscapes:
```
export FINE_ANN=<path to fine annotations>
export IMG_DIR=<path to leftImg8bit_transmittance>
python3 eval-cityscapes.py
```

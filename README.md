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
python3 eval-bdd100k.py --paths <path to validation set> <path to labels/det_val.json>
```

Evaluate on the Mapillary validation set:
```
python3 eval-mapillary.py --paths <path to validation set> <path to v2.0/polygons>
```

To evaluate the model on the Indian Driving Dataset, first create the validation set:
```
python3 make-idd-eval.py --paths <path to val.txt> <path to annotations directory> <path to dataset directory> <path to IDD>
```
Then, run the evaluation script:
```
python3 eval-idd.py --paths <path to eval-img>
```

Evaluate on Foggy Cityscapes (with attenuation coefficient of 0.02):
```
python3 eval-foggy-cityscapes.py --paths <path to fine annotations> <path to leftImg8bit_transmittance>
```

Evaluate on Cityscapes:
```
python3 eval-cityscapes.py --paths <path to fine annotations> <path to leftImg8bit_transmittance>
```
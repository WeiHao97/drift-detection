
# Drift Detection

First, create a virtual environment and install all dependencies:
``` 
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```
To run a streaming simulation on the office dataset (without online training), invoke the following:

`python3 office_simulator.py <path to reference samples> <path to first set of target samples> <path to second set of target samples> <trained model> <test set filenames for reference distribution>`
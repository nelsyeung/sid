# Sid (Salt Identification)

## Getting started
### Installation
Make sure you have Python 3.5 installed, Python 3.6+ isn't officially supported
for Tensorflow.

Create a virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate
```

Install required libraries:
```
pip install -r requirements.txt
```

Install required libraries for development:
```
pip install -r requirements-dev.txt
```

Download [all data](https://www.kaggle.com/c/tgs-salt-identification-challenge/data)
and unzip the files into the `input` folder. Unzip the `train.zip` and
`test.zip` to `input/train` and `input/test`, respectively.

### Train
Use the train script to train data inside `input/train`:
```
./train.py
```
It will write a model file `model.h5`.

### Predict
Use the predict script which reads in `model.h5` to produce a `submission.csv`
file:
```
./predict.py
```

### Progress bar
To log progress for different part of the scripts, set the `PROGRESS`
environment variable:
```
PROGRESS=1 ./train.py
PROGRESS=1 ./predict.py
```

Note that `PROGRESS` can be set to anything and progress will be logged. For
example, all of below will log progress:
```
PROGRESS=False ./train.py
PROGRESS=0 ./train.py
```

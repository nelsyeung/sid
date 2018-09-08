# Sid (Salt Identification)

## Getting started
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

Use the train script to train data inside `input/train`:
```
python train.py
```
It will write a model file `model-sid.h5`, then execute the predict script to
produce a `submission.csv` file:
```
python predict.py
```

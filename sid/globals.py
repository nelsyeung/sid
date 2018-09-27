import os

width = 224
height = 224
channels = 3
file_model = 'model.h5'
seed = int(os.environ['SEED']) if 'SEED' in os.environ else 1
progress = True if 'PROGRESS' in os.environ else False

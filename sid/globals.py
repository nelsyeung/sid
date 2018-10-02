import os

file_model = 'model.h5'
width = 224
height = 224
channels = 3
progress = True if 'PROGRESS' in os.environ else False
debug = True if 'DEBUG' in os.environ else False
debug_dir = 'debug'
seed = int(os.environ['SEED']) if 'SEED' in os.environ else 1

import os

file_model = 'model.h5'
width = 128
height = 128
channels = 1
debug = True if 'DEBUG' in os.environ else False
debug_dir = 'debug'
seed = int(os.environ['SEED']) if 'SEED' in os.environ else 1
import os
import sys
import shutil


task = sys.argv[1]
experiment = sys.argv[2]

weights = '/home/weights/task'
tensorboard = '/home/tensorboard/task'

try:
    shutil.rmtree(os.path.join(weights+str(task), experiment))
except:
    pass
try:
    shutil.rmtree(os.path.join(tensorboard+str(task), experiment))
except:
    pass

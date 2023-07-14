import os
import json
import numpy as np
import matplotlib.pylab as plt

from tensorflow.keras.models import load_model

from planktonclas.data_utils import load_image, load_data_splits, load_class_names
from planktonclas.test_utils import predict
from planktonclas import paths, plot_utils, utils
# 2023-06-05_143422
# User parameters to set
TIMESTAMP = '2023-06-05_143422'                       # timestamp of the model
MODEL_NAME = 'final_model.h5'                           # model to use to make the prediction
TOP_K = 5                                               # number of top classes predictions to save

# Set the timestamp
paths.timestamp = TIMESTAMP

# Load the data
class_names = load_class_names(splits_dir=paths.get_ts_splits_dir())

# Load training configuration
conf_path = os.path.join(paths.get_conf_dir(), 'conf.json')
with open(conf_path) as f:
    conf = json.load(f)
    
# Load the model
model = load_model(os.path.join(paths.get_checkpoints_dir(), MODEL_NAME), custom_objects=utils.get_custom_objects())


FILEPATHS = ['/media/ignacio/Datos/datasets/semillas/datasets/RJB/Euphorbia_terracina_JC1355_SEM_COL.jpg',
             '/media/ignacio/Datos/datasets/semillas/datasets/RJB/Campanula_lusitanica_lusitanica_LM4461_SEM_COL.jpg',
             '/media/ignacio/Datos/datasets/semillas/datasets/RJB/Arbutus_unedo_RJB03_1_COL.jpg']
     
pred_lab, pred_prob = predict(model, FILEPATHS, conf, top_K=TOP_K, filemode='local')

for i, im_path in enumerate(FILEPATHS):
    plt.figure(i)
    plt.imshow(load_image(im_path, filemode='local'))
    plt.show()
    for j in range(pred_lab.shape[1]):
        print('[{:.1f}%] {}'.format(pred_prob[i, j] * 100, class_names[pred_lab[i, j]]))
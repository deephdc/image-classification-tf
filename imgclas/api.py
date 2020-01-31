"""
API for the image classification package

Date: September 2018
Author: Ignacio Heredia
Email: iheredia@ifca.unican.es
Github: ignacioheredia

Notes: Based on https://github.com/indigo-dc/plant-classification-theano/blob/package/plant_classification/api.py

Descriptions:
The API will use the model files inside ../models/api. If not found it will use the model files of the last trained model.
If several checkpoints are found inside ../models/api/ckpts we will use the last checkpoint.

Warnings:
There is an issue of using Flask with Keras: https://github.com/jrosebr1/simple-keras-rest-api/issues/1
The fix done (using tf.get_default_graph()) will probably not be valid for standalone wsgi container e.g. gunicorn,
gevent, uwsgi.
"""

import json
import os
import warnings
from datetime import datetime
import pkg_resources
import builtins
import re
from collections import OrderedDict

import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from webargs import fields
from aiohttp.web import HTTPBadRequest

from imgclas import paths, utils, config, test_utils
from imgclas.data_utils import load_class_names, load_class_info, mount_nextcloud
from imgclas.train_runfile import train_fn


# TODO: Move to proper marshalling for arguments
# The point is that some fields need additional information than the one that is contained in the config.yaml
# --> define an example for each arg in the config.yaml --> create the schema for that arg
# type_map = {'int': fields.Int, 'str': fields.Str, 'float': fields.Float, 'dict': fields.Dict,
#             'bool': fields.Bool, 'list': fields.List}
# field_type = type_map.get(g_val['type'], fields.Field)
# parser[g_key] = field_type(**opt_args)
# --> another option is to add a marshmallow schema to each config args


# Mount NextCloud folders (if NextCloud is available)
try:
    mount_nextcloud('rshare:/data/dataset_files', paths.get_splits_dir())
    mount_nextcloud('rshare:/data/images', paths.get_images_dir())
    #mount_nextcloud('rshare:/models', paths.get_models_dir())
except Exception as e:
    print(e)

# Empty model variables for inference (will be loaded the first time we perform inference)
loaded_ts, loaded_ckpt = None, None
graph, model, conf, class_names, class_info = None, None, None, None, None

# Additional parameters
allowed_extensions = set(['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']) # allow only certain file extensions
top_K = 5  # number of top classes predictions to return


def load_inference_model(timestamp=None, ckpt_name=None):
    """
    Load a model for prediction.

    Parameters
    ----------
    * timestamp: str
        Name of the timestamp to use. The default is `api` or the last timestamp in `./models` if `api` is not
        available.
    * ckpt_name: str
        Name of the checkpoint to use. The default is `final_model.h5` or the last checkpoint in
        `./models/[timestamp]/ckpts` if `final_model.h5` is not available.
    """
    global loaded_ts, loaded_ckpt
    global graph, model, conf, class_names, class_info

    # Set the timestamp
    timestamp_list = next(os.walk(paths.get_models_dir()))[1]
    timestamp_list = sorted(timestamp_list)
    if not timestamp_list:
        raise Exception(
            "You have no models in your `./models` folder to be used for inference. "
            "Therefore the API can only be used for training.")
    elif timestamp is None:
        timestamp = timestamp_list[-1]
    elif timestamp not in timestamp_list:
        raise ValueError(
            "Invalid timestamp name: {}. Available timestamp names are: {}".format(timestamp, timestamp_list))
    paths.timestamp = timestamp
    print('Using TIMESTAMP={}'.format(timestamp))

    # Set the checkpoint model to use to make the prediction
    ckpt_list = os.listdir(paths.get_checkpoints_dir())
    ckpt_list = sorted([name for name in ckpt_list if name.endswith('.h5')])
    if not ckpt_list:
        raise Exception(
            "You have no checkpoints in your `./models/{}/ckpts` folder to be used for inference. ".format(timestamp) +
            "Therefore the API can only be used for training.")
    elif ckpt_name is None:
        ckpt_name = ckpt_list[-1]
    elif ckpt_name not in ckpt_list:
        raise ValueError(
            "Invalid checkpoint name: {}. Available checkpoint names are: {}".format(ckpt_name, ckpt_list))
    print('Using CKPT_NAME={}'.format(ckpt_name))

    # Clear the previous loaded model
    K.clear_session()

    # Load the class names and info
    splits_dir = paths.get_ts_splits_dir()
    class_names = load_class_names(splits_dir=splits_dir)
    class_info = None
    if 'info.txt' in os.listdir(splits_dir):
        class_info = load_class_info(splits_dir=splits_dir)
        if len(class_info) != len(class_names):
            warnings.warn("""The 'classes.txt' file has a different length than the 'info.txt' file.
            If a class has no information whatsoever you should leave that classes row empty or put a '-' symbol.
            The API will run with no info until this is solved.""")
            class_info = None
    if class_info is None:
        class_info = ['' for _ in range(len(class_names))]

    # Load training configuration
    conf_path = os.path.join(paths.get_conf_dir(), 'conf.json')
    with open(conf_path) as f:
        conf = json.load(f)
        update_with_saved_conf(conf)

    # Load the model
    model = load_model(os.path.join(paths.get_checkpoints_dir(), ckpt_name),
                       custom_objects=utils.get_custom_objects())
    graph = tf.get_default_graph()

    # Set the model as loaded
    loaded_ts = timestamp
    loaded_ckpt = ckpt_name


def update_with_saved_conf(saved_conf):
    """
    Update the default YAML configuration with the configuration saved from training
    """
    # Update the default conf with the user input
    CONF = config.CONF
    for group, val in sorted(CONF.items()):
        if group in saved_conf.keys():
            for g_key, g_val in sorted(val.items()):
                if g_key in saved_conf[group].keys():
                    g_val['value'] = saved_conf[group][g_key]

    # Check and save the configuration
    config.check_conf(conf=CONF)
    config.conf_dict = config.get_conf_dict(conf=CONF)


def update_with_query_conf(user_args):
    """
    Update the default YAML configuration with the user's input args from the API query
    """
    # Update the default conf with the user input
    CONF = config.CONF
    for group, val in sorted(CONF.items()):
        for g_key, g_val in sorted(val.items()):
            if g_key in user_args:
                g_val['value'] = json.loads(user_args[g_key])

    # Check and save the configuration
    config.check_conf(conf=CONF)
    config.conf_dict = config.get_conf_dict(conf=CONF)


def catch_error(f):
    def wrap(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            raise HTTPBadRequest(reason=e)
    return wrap


def catch_url_error(url_list):

    # Error catch: Empty query
    if not url_list:
        raise ValueError('Empty query')

    for i in url_list:
        if not i.startswith('data:image'):  # don't do the checks for base64 encoded images

            # Error catch: Inexistent url
            try:
                url_type = requests.head(i).headers.get('content-type')
            except Exception:
                raise ValueError("Failed url connection: "
                                 "Check you wrote the url address correctly.")

            # Error catch: Wrong formatted urls
            if url_type.split('/')[0] != 'image':
                raise ValueError("Url image format error: Some urls were not in image format. "
                                 "Check you didn't uploaded a preview of the image rather than the image itself.")


def catch_localfile_error(file_list):

    # Error catch: Empty query
    if not file_list:
        raise ValueError('Empty query')

    # Error catch: Image format error
    for f in file_list:
        extension = os.path.basename(f.content_type).split('/')[-1]
        # extension = mimetypes.guess_extension(f.content_type)
        if extension not in allowed_extensions:
            raise ValueError("Local image format error: "
                             "At least one file is not in a standard image format ({}).".format(allowed_extensions))


def warm():
    load_inference_model()


@catch_error
def predict(**args):

    if (not any([args['urls'], args['files']]) or
            all([args['urls'], args['files']])):
        raise Exception("You must provide either 'url' or 'data' in the payload")

    if args['files']:
        args['files'] = [args['files']]  # patch until list is available
        return predict_data(args)
    elif args['urls']:
        args['urls'] = [args['urls']]  # patch until list is available
        return predict_url(args)


def predict_url(args):
    """
    Function to predict an url
    """
    # Check user configuration
    update_with_query_conf(args)
    conf = config.conf_dict

    merge = True
    catch_url_error(args['urls'])

    # Load model if needed
    if loaded_ts != conf['testing']['timestamp'] or loaded_ckpt != conf['testing']['ckpt_name']:
        load_inference_model(timestamp=conf['testing']['timestamp'],
                             ckpt_name=conf['testing']['ckpt_name'])
        conf = config.conf_dict

    # Make the predictions
    with graph.as_default():
        pred_lab, pred_prob = test_utils.predict(model=model,
                                                 X=args['urls'],
                                                 conf=conf,
                                                 top_K=top_K,
                                                 filemode='url',
                                                 merge=merge,
                                                 use_multiprocessing=False)  # safer to avoid memory fragmentation in failed queries

    if merge:
        pred_lab, pred_prob = np.squeeze(pred_lab), np.squeeze(pred_prob)

    return format_prediction(pred_lab, pred_prob)


def predict_data(args):
    """
    Function to predict an image in binary format
    """
    # Check user configuration
    update_with_query_conf(args)
    conf = config.conf_dict

    merge = True
    catch_localfile_error(args['files'])

    # Load model if needed
    if loaded_ts != conf['testing']['timestamp'] or loaded_ckpt != conf['testing']['ckpt_name']:
        load_inference_model(timestamp=conf['testing']['timestamp'],
                             ckpt_name=conf['testing']['ckpt_name'])
        conf = config.conf_dict

    # Create a list with the path to the images
    filenames = [f.filename for f in args['files']]

    # Make the predictions
    try:
        with graph.as_default():
            pred_lab, pred_prob = test_utils.predict(model=model,
                                                     X=filenames,
                                                     conf=conf,
                                                     top_K=top_K,
                                                     filemode='local',
                                                     merge=merge,
                                                     use_multiprocessing=False)  # safer to avoid memory fragmentation in failed queries
    finally:
        for f in filenames:
            os.remove(f)

    if merge:
        pred_lab, pred_prob = np.squeeze(pred_lab), np.squeeze(pred_prob)

    return format_prediction(pred_lab, pred_prob)


def format_prediction(labels, probabilities):
    d = {
        "status": "ok",
        "predictions": [],
    }

    for label_id, prob in zip(labels, probabilities):
        name = class_names[label_id]

        pred = {
            "label_id": int(label_id),
            "label": name,
            "probability": float(prob),
            "info": {
                "links": {'Google images': image_link(name),
                          'Wikipedia': wikipedia_link(name)},
                'metadata': class_info[label_id],
            },
        }
        d["predictions"].append(pred)
    return d


def image_link(pred_lab):
    """
    Return link to Google images
    """
    base_url = 'https://www.google.es/search?'
    params = {'tbm':'isch','q':pred_lab}
    link = base_url + requests.compat.urlencode(params)
    return link


def wikipedia_link(pred_lab):
    """
    Return link to wikipedia webpage
    """
    base_url = 'https://en.wikipedia.org/wiki/'
    link = base_url + pred_lab.replace(' ', '_')
    return link


def train(**args):
    """
    Train an image classifier
    """
    update_with_query_conf(user_args=args)
    CONF = config.conf_dict
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    config.print_conf_table(CONF)
    K.clear_session()  # remove the model loaded for prediction
    train_fn(TIMESTAMP=timestamp, CONF=CONF)
    
    # Sync with NextCloud folders (if NextCloud is available)
    try:
        mount_nextcloud(paths.get_models_dir(), 'rshare:/models')
    except Exception as e:
        print(e)    


def populate_parser(parser, default_conf):
    """
    Returns a arg-parse like parser.
    """
    for group, val in default_conf.items():
        for g_key, g_val in val.items():
            gg_keys = g_val.keys()

            # Load optional keys
            help = g_val['help'] if ('help' in gg_keys) else ''
            type = getattr(builtins, g_val['type']) if ('type' in gg_keys) else None
            choices = g_val['choices'] if ('choices' in gg_keys) else None

            # Additional info in help string
            help += '\n' + "<font color='#C5576B'> Group name: **{}**".format(str(group))
            if choices:
                help += '\n' + "Choices: {}".format(str(choices))
            if type:
                help += '\n' + "Type: {}".format(g_val['type'])
            help += "</font>"

            # Create arg dict
            opt_args = {'missing': json.dumps(g_val['value']),
                        'description': help,
                        'required': False,
                        }
            if choices:
                opt_args['enum'] = [json.dumps(i) for i in choices]

            parser[g_key] = fields.Str(**opt_args)

    return parser


def get_train_args():

    parser = OrderedDict()
    default_conf = config.CONF
    default_conf = OrderedDict([('general', default_conf['general']),
                                ('model', default_conf['model']),
                                ('training', default_conf['training']),
                                ('monitor', default_conf['monitor']),
                                ('dataset', default_conf['dataset']),
                                ('augmentation', default_conf['augmentation'])])

    return populate_parser(parser, default_conf)


def get_predict_args():

    parser = OrderedDict()
    default_conf = config.CONF
    default_conf = OrderedDict([('testing', default_conf['testing'])])

    # Add options for modelname
    timestamp = default_conf['testing']['timestamp']
    timestamp_list = next(os.walk(paths.get_models_dir()))[1]
    timestamp_list = sorted(timestamp_list)
    if not timestamp_list:
        timestamp['value'] = ''
    else:
        timestamp['value'] = timestamp_list[-1]
        timestamp['choices'] = timestamp_list

    # Add data and url fields
    parser['files'] = fields.Field(required=False,
                                   missing=None,
                                   type="file",
                                   data_key="data",
                                   location="form",
                                   description="Select the image you want to classify.")

    # Use field.String instead of field.Url because I also want to allow uploading of base 64 encoded data strings
    parser['urls'] = fields.String(required=False,
                                   missing=None,
                                   description="Select an URL of the image you want to classify.")

    # missing action="append" --> append more than one url

    return populate_parser(parser, default_conf)


def get_metadata(distribution_name='imgclas'):
    """
    Function to read metadata
    """

    pkg = pkg_resources.get_distribution(distribution_name)
    meta = {
        'Name': None,
        'Version': None,
        'Summary': None,
        'Home-page': None,
        'Author': None,
        'Author-email': None,
        'License': None,
    }

    for line in pkg.get_metadata_lines("PKG-INFO"):
        for par in meta:
            if line.startswith(par):
                _, value = line.split(": ", 1)
                meta[par] = value

    # Update information with Docker info (provided as 'CONTAINER_*' env variables)
    r = re.compile("^CONTAINER_(.*?)$")
    container_vars = list(filter(r.match, list(os.environ)))
    for var in container_vars:
        meta[var.capitalize()] = os.getenv(var)

    return meta

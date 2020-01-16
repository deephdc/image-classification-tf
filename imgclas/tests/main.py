"""
Gather all module's test

Date: December 2019
Author: Ignacio Heredia
Email: iheredia@ifca.unican.es
Github: ignacioheredia
"""
import os
import subprocess
import time
import json
from shutil import copyfile

from imgclas import paths


module_name = 'imgclas'
test_url = 'https://file-examples.com/wp-content/uploads/2017/10/file_example_JPG_100kB.jpg'


# ===========
# Local Tests
# ===========

def test_load():
    print('Testing local: module load ...')
    import imgclas.api


def test_metadata():
    print('Testing local: metadata ...')
    from imgclas.api import get_metadata

    get_metadata()


def test_predict_url():
    print('Testing local: predict url ...')
    from imgclas.api import predict_url

    args = {'urls': [test_url]}
    r = predict_url(args)


def test_predict_data():
    print('Testing local: predict data ...')
    from deepaas.model.v2.wrapper import UploadedFile
    from imgclas.api import predict_data

    fpath = os.path.join(paths.get_base_dir(), 'data', 'samples', 'sample.jpg')
    tmp_fpath = os.path.join(paths.get_base_dir(), 'data', 'samples', 'tmp_file.jpg')
    copyfile(fpath, tmp_fpath)  # copy to tmp because we are deleting the file after prediction
    file = UploadedFile(name='data', filename=tmp_fpath, content_type='image/jpg')
    args = {'files': [file]}
    r = predict_data(args)


# ==========
# CURL Tests
# ==========

def test_curl_load():
    print('Testing curl: module load ...')

    r = subprocess.run('curl -X GET "http://0.0.0.0:5000/v2/models/" -H "accept: application/json"',
                       shell=True, check=True, stdout=subprocess.PIPE).stdout
    r = json.loads(r)
    models = [m['name'] for m in r['models']]
    if module_name not in models:
        raise Exception('Model is not correctly loaded.')


def test_curl_metadata():
    print('Testing curl: metadata ...')

    r = subprocess.run('curl -X GET "http://0.0.0.0:5000/v2/models/{}/" -H "accept: application/json"'.format(module_name),
                       shell=True, check=True, stdout=subprocess.PIPE).stdout
    if r == b'404: Not Found':
        raise Exception('Model is not correctly loaded.')
    r = json.loads(r)


def test_curl_predict_url():
    print('Testing curl: predict url ...')
    from urllib.parse import quote_plus

    r = subprocess.run('curl -X POST "http://0.0.0.0:5000/v2/models/{}/predict/?urls={}" -H "accept: application/json"'.format(module_name,
                                                                                                                               quote_plus(test_url)),                       shell=True, check=True, stdout=subprocess.PIPE).stdout
    if r == b'404: Not Found':
        raise Exception('Model is not correctly loaded.')
    r = json.loads(r)


if __name__ == '__main__':
    print('Testing locally ...')
    test_load()
    test_metadata()
    test_predict_url()
    test_predict_data()

    print('Testing through CURL ...')
    r = subprocess.run('deepaas-run --listen-ip 0.0.0.0 --nowarm &', shell=True)  # launch deepaas
    time.sleep(20)  # wait for deepaas to be ready
    test_curl_load()
    test_curl_metadata()
    test_curl_predict_url()
    r = subprocess.run("kill $(ps aux | grep 'deepaas-run' | awk '{print $2}')", shell=True)   # kill deepaas

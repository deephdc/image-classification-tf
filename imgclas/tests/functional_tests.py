"""
File to run some unit test on the API
"""
import os

from deepaas.model.v2.wrapper import UploadedFile

from imgclas import paths
from imgclas.api import predict_data, predict_url


def test_predict_url():
    url = 'https://file-examples.com/wp-content/uploads/2017/10/file_example_JPG_100kB.jpg'
    args = {'urls': [url]}
    results = predict_url(args)
    # print(results)


def test_predict_data():
    fpath = os.path.join(paths.get_base_dir(), 'data', 'samples', 'sample.jpg')
    file = UploadedFile(name='data', filename=fpath, content_type='image/jpg')
    args = {'files': [file]}
    results = predict_data(args)
    # print(results)


if __name__ == '__main__':
    # test_predict_data()
    # test_predict_url()

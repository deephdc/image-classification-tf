"""
File to run some unit test on the API
"""
import os

from werkzeug.datastructures import FileStorage

from imgclas import paths
from imgclas.api import predict_data, predict_url


def test_predict_url():
    url = 'https://file-examples.com/wp-content/uploads/2017/10/file_example_JPG_100kB.jpg'
    args = {'urls': [url]}
    results = predict_url(args)


def test_predict_data():
    fpath = os.path.join(paths.get_base_dir(), 'data', 'samples', 'sample.jpg')
    file = FileStorage(open(fpath, 'rb'))
    args = {'files': file}
    results = predict_data(args)


if __name__ == '__main__':
    test_predict_data()
    test_predict_url()

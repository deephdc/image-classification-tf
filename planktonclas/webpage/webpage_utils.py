"""
Utils for the image classification webpage

Date: September 2018
Author: Ignacio Heredia
Email: iheredia@ifca.unican.es
Github: ignacioheredia
"""

import os

from jinja2 import Template

homedir = os.path.dirname(os.path.abspath(__file__))

def create_labels_html(labels):
    """
    Create a labels.html file from a list of label names
    """

    with open(os.path.join(homedir, 'templates', 'labels_template.html'), 'r') as f:
        template = Template(f.read())

    custom = template.stream(labels=labels)
    custom.dump(os.path.join('templates', 'labels.html'))


def filestorage_to_binary(file_list):
    """
    Transform a list of images in Flask's FileStorage format to binary format.
    """
    binary_list = []
    for f in file_list:
        binary_list.append(f.read())
    return binary_list

"""
Image classification webpage

Date: September 2018
Author: Ignacio Heredia
Email: iheredia@ifca.unican.es
Github: ignacioheredia

Description:
This script launches a basic webpage interface to return results for image classification prediction.
To launch this webpage you can run `python serve.py`.

Tip:
To host the app in a subpath through a proxy_pass with nginx check Ross's anwer in [1].
Redirections must then be made with either:
* redirect(url_for('intmain', _external=True))    
* redirect('./')

References:
[1] https://stackoverflow.com/questions/25962224/running-a-flask-application-at-a-url-that-is-not-the-domain-root
"""

import os

from flask import Flask, render_template, request, send_from_directory, json, Response, flash, Markup

from imgclas import api
from imgclas.webpage import webpage_utils


# Configuration parameters of the web application
app = Flask(__name__)
if os.path.isfile('secret_key.txt'):
    app.secret_key = open('secret_key.txt', 'r').read()
else:
    app.secret_key = 'devkey, should be in a file'

# Load model
if not api.loaded:
    api.load_inference_model()

# Create labels.html from synsets.txt
webpage_utils.create_labels_html(labels=api.class_names)

print('Ready!')

@app.route('/')
def intmain():
    return render_template('index.html')


@app.route('/labels')
def label_list():
    return render_template('labels.html')


@app.route('/robots.txt')
def static_from_root():
    return send_from_directory(app.static_folder, request.path[1:])


@app.route('/url_upload', methods=['POST'])
def url_post():
    url_list = request.form['url']
    url_list = [i.replace(' ', '') for i in url_list.split(' ') if i != '']
    args = {'urls': url_list}

    try:
        message = api.predict_url(args, merge=True)
    except Exception as error:
        print(error)
        flash(Markup(error))
        code = error.code if hasattr(error, 'code') else 500
        return render_template('index.html'), code

    return render_template('results.html', predictions=message['predictions'])


@app.route('/local_upload', methods=['POST'])
def local_post():
    uploaded_files = request.files.getlist('local_files')
    # uploaded_files = webpage_utils.filestorage_to_binary(uploaded_files)
    args = {'files': uploaded_files}

    try:
        message = api.predict_data(args)
    except Exception as error:
        print(error)
        flash(Markup(error))
        code = error.code if hasattr(error, 'code') else 500
        return render_template('index.html'), code

    return render_template('results.html', predictions=message['predictions'])


@app.route('/api', methods=['POST'])
def api_fn():

    mode = request.form.get('mode')
    if mode == 'url':
        im_list = request.form.getlist('url_list')
        message = api.predict_url(im_list, merge=True)
    elif mode == 'localfile':
        im_list = request.files.to_dict().values()
        im_list = webpage_utils.filestorage_to_binary(im_list)
        message = api.predict_data(images=im_list)
    else:
        message = {'status': 'error', 'Error_type': 'Invalid mode'}

    js = json.dumps(message)
    if message['status'] == 'ok':
        resp = Response(js, status=200, mimetype='application/json')
    if message['status'] == 'error':
        resp = Response(js, status=400, mimetype='application/json')

    return resp


@app.errorhandler(404)
def page_not_found(e):
    flash(Markup(e))
    return render_template('index.html'), 404


@app.errorhandler(405)
def method_not_allowed(e):
    flash(Markup(e))
    return render_template('index.html'), 405


if __name__ == '__main__':
    app.debug = False
    app.run()

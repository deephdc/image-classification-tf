# Webpage docs

This is a basic web app to implement image recognition for **predicting** with local files and urls.

## Preliminaries

You have to do some preliminary steps to select the model you want to predict with:

- copy your desired  `.models/[timestamp]` to `.models/api`. If there is no `.models/api` folder, the default is to use the last available timestamp.
- in the `.models/api/ckpts` leave only the desired checkpoint to use for prediction. If there are more than one chekpoints, the default is to use the last available checkpoint.


## Launching the webpage

To launch the web execute the following:

```bash
cd ./imgclas/webpage
python serve.py
```
and it will start running at http://127.0.0.1:5000. To run the webpage in production mode you can pip-install the `gunicorn` module as an easy drop-in replacement. Once installed just run

```bash
cd ./imgclas/webpage
gunicorn serve:app -b 0.0.0.0:80 --workers 1 --timeout 80 -k gevent
```

Beware of the number of workers because you might get `OUT_OF_MEMORY` errors if using gpu.

## Using the API

You can query your webpage also through an API. You have to make a POST request with the images belonging to your observation.

### Python snippets
Here are some Python snippets using the [requests](https://pypi.org/project/requests/) module.

**Classifying URLs**
```python
im_list = ['https://public-media.smithsonianmag.com/filer/89/47/8947cd5c-ac01-4c0e-891a-505517cc0663/istock-540753808.jpg', 
           'https://cdn.pixabay.com/photo/2014/04/10/11/24/red-rose-320868_960_720.jpg']

r = requests.post('http://127.0.0.1:5000/api', data={'mode':'url', 'url_list':im_list})
```

**Classifying local images**

```python
im_paths = ['/home/ignacio/image_recognition/data/demo-images/image1.jpg',
            '/home/ignacio/image_recognition/data/demo-images/image2.jpg']

im_names = [str(i) for i in range(len(im_paths))]
im_files = [open(f, 'rb') for f in im_paths]
file_dict = dict(zip(im_names, im_files))

r = requests.post('http://127.0.0.1:5000/api', data={'mode':'localfile'}, files=file_dict)
```

### CURL snippets

**Classifying URLs**
```bash
curl --data "mode=url&url_list=https://public-media.smithsonianmag.com/filer/89/47/8947cd5c-ac01-4c0e-891a-505517cc0663/istock-540753808.jpg&url_list=https://cdn.pixabay.com/photo/2014/04/10/11/24/red-rose-320868_960_720.jpg" http://127.0.0.1:5000/api
```

**Classifying local images**
```bash
curl --form mode=localfile --form 0=@/home/ignacio/image_recognition/data/demo-images/image1.jpg --form 1=@/home/ignacio/image_recognition/data/demo-images/image2.jpg http://deep.ifca.es/api
```

### Responses

A successful response should return a json, with the labels and their respective probabilities, like the following

```python
{'predictions': [{'info': {'links': {'Google images': 'https://www.google.es/search?tbm=isch&q=Genus+Chenopodium',
                                     'Wikipedia': 'https://en.wikipedia.org/wiki/Genus_Chenopodium'},
                           'metadata': '516 images in DB'},
                  'label': 'Genus Chenopodium',
                  'label_id': 118,
                  'probability': 0.01801883988082409},
                  
                 {'info': {'links': {'Google images': 'https://www.google.es/search?tbm=isch&q=Genus+Rumex',
                                     'Wikipedia': 'https://en.wikipedia.org/wiki/Genus_Rumex'},
                           'metadata': '456 images in DB'},
                  'label': 'Genus Rumex',
                  'label_id': 409,
                  'probability': 0.016028326004743576},
                  
                 {'info': {'links': {'Google images': 'https://www.google.es/search?tbm=isch&q=Genus+Sinapis',
                                     'Wikipedia': 'https://en.wikipedia.org/wiki/Genus_Sinapis'},
                           'metadata': '394 images in DB'},
                  'label': 'Genus Sinapis',
                  'label_id': 435,
                  'probability': 0.013632828369736671},
                  
                 {'info': {'links': {'Google images': 'https://www.google.es/search?tbm=isch&q=Genus+Carex',
                                     'Wikipedia': 'https://en.wikipedia.org/wiki/Genus_Carex'},
                           'metadata': '305 images in DB'},
                  'label': 'Genus Carex',
                  'label_id': 104,
                  'probability': 0.010641136206686497},
                  
                 {'info': {'links': {'Google images': 'https://www.google.es/search?tbm=isch&q=Genus+Sorghum',
                                     'Wikipedia': 'https://en.wikipedia.org/wiki/Genus_Sorghum'},
                           'metadata': '302 images in DB'},
                  'label': 'Genus Sorghum',
                  'label_id': 443,
                  'probability': 0.010484464466571808}],
                  
 'status': 'ok'}
```

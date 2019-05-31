import cv2
from flask import Flask, request
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
import tensorflow as tf
import numpy as np
import flask

app = Flask(__name__)
model = load_model('iv3-deployed.model')
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
classes = ['cat', 'dog']

img_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

graph = tf.get_default_graph()

# request model prediction
@app.route('/classify', methods=['POST'])
def classify():
    # read image from the request
    req_img = request.files['image'].read()
    nparr = np.fromstring(req_img, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # format the image to fit the Input of the model
    preprocessed = img_gen.standardize(img)
    formatted = np.expand_dims(preprocessed, 0)

    # Required because of a bug in Keras when using tensorflow graph across threads
    with graph.as_default():
        result = model.predict(formatted)
        result = np.argmax(result)
        data = {'result': classes[result]}
        return flask.jsonify(data)


# start Flask server
app.run(host='0.0.0.0', port=22666, debug=False)

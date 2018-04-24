import numpy as np 
import os as os, os.path
import sys
import re
# sys.path.append(os.path.abspath('./model'))
from scipy.misc.pilutil import imsave, imread, imresize
from PIL import Image
from flask import (Flask, request, g, redirect, url_for, abort, Response, jsonify)
from flask_cors import CORS
import base64
import tensorflow as tf


"""
Import all the dependencies you need to load the model, 
preprocess your request and postprocess your result
"""
app = Flask(__name__)
CORS(app) # needed for cross-domain requests, allow everything by default
MODEL_PATH = os.getcwd() + '/data/'



# def load_model(MODEL_PATH):
#     """Load the model"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

sess = tf.Session('', tf.Graph())
with sess.graph.as_default():       
    saver = tf.train.import_meta_graph(MODEL_PATH + "trained_model.ckpt.meta")
    saver.restore(sess, MODEL_PATH + "trained_model.ckpt")

        # Get pointers to relevant tensors in graph
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("X_placeholder:0") # input
    y = graph.get_tensor_by_name("Y_placeholder:0") # label - not using this, unless we want to calculate loss
    is_training = graph.get_tensor_by_name( "is_training:0" ) # have to feed False to make sure batch norm and dropout behaves accordingly
    prediction = graph.get_tensor_by_name( "Prediction:0" ) # these will be results


def data_preprocessing(data):
    """Preprocess the request data to transform it to a format that the model understands"""
    imgstr = re.search(b'base64,(.*)',data).group(1)
    with open('output.jpg','wb') as output:
         output.write(base64.b64decode(imgstr))
    
# Every incoming POST request will run the `evaluate` method
# The request method is POST (this method enables your to send arbitrary data to the endpoint in the request body, including images, JSON, encoded-data, etc.)
@app.route('/api', methods=["POST"])
def evaluate():

    print('Got to evaluate...')

    imageData = request.get_data()
    # CODE FOR DATA PREPROCESSING
    data_preprocessing(imageData)
    print('1: Image was converted')

    image = Image.open('output.jpg')
    image = image.convert('L')
    image = image.resize((50, 50), Image.ANTIALIAS)
    print('Resized and Grayscaled Image') 

    image = np.expand_dims(np.array(image), axis = 0)
    classification = sess.run(prediction, feed_dict = {x: image, is_training : True})   
    classes = np.argmax( classification, axis = 1 ) # add highest probability result to classes

    res = 'undefined'
    if   classes[0] == 0:
        print('Predicted: Angry')
        res = 'Angry'
    elif classes[0] == 1:
        print('Predicted: Fear')
        res = 'Fear'
    elif classes[0] == 2:
        print('Predicted: Happy')
        res = 'Happy'
    elif classes[0] == 3:
        print('Predicted: neutral')
        res = 'Neutral'
    elif classes[0] == 4:
        print('Predicted: sad')
        res = 'Sad'
    elif classes[0] == 5:
        print('Predicted: Suprised')
        res = 'Suprised'

    return jsonify(res)
@app.route('/api', methods=["GET"])
def api():

    return 'Get Request Recieved...'




# Load the model and run the server
if __name__ == "__main__":
    print(("* Loading model and starting Flask server..."
        "please wait until server has fully started"))
    app.debug = True
    # load_model(MODEL_PATH)
    app.run(host='0.0.0.0')
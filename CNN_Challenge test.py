# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:00:59 2024

@author: Adrian Ho
"""

# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(32, 32))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 32, 32, 3)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img
 
# load an image and predict the class
def run_example():
	# load the image
	input_filename = input("Input file number:")
	img = load_image(input_filename)
	#class name
	class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
	# load model
	model = load_model('final_model2(90.4).h5')
	# predict the class
	predictions = (model.predict(img) > 0.5).astype("int32")
	a = predictions.flatten()
	res = dict(zip(a, class_names))
	print(res.get(1))

# entry point, run the example
run_example()
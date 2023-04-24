"""
Gorspy v2
	- build model on one image at a time
	- recreate each image with model
	- stitch images together
"""

from PIL import Image
import numpy as np
from joblib import Parallel, delayed
import pickle
import os
import os.path


def main():
	# map a red, green, and blue model to each image
	# image_and_models = [{
	# 	'image_data': image_data,
	# 	'split_data': split_image_data(image_data),
	# 	'red_model': LinearRegression(),
	# 	'green_model': LinearRegression(),
	# 	'blue_model': LinearRegression()
	# } for image_data in original_images]

	# process_1()
	process_2()
	# process_3()


def process_1():
	"""
	original process: train models on source images, test model on stitched image
	:return: void
	"""
	# initialize the testing image
	print('loading test image...')
	test_mountain_image = Image.open('mountain_images_testing/phoenix-mountain-preserve-short.jpg')
	test_mountain_image_data = np.asarray(test_mountain_image)
	test_mountain_image_data_normal = normalize(test_mountain_image_data)
	test_data_split = split_image_data(test_mountain_image_data_normal)
	print('test image loaded')

	# initialize models for each color channel
	# TODO: shorten this if statement
	if os.path.isfile('preloaded_models/red_model.pkl') and \
			os.path.isfile('preloaded_models/green_model.pkl') and \
			os.path.isfile('preloaded_models/blue_model.pkl'):
		print('loading preloaded models...')
		with open('preloaded_models/red_model.pkl', 'rb') as red_file:
			red_model = pickle.load(red_file)
		with open('preloaded_models/green_model.pkl', 'rb') as green_file:
			green_model = pickle.load(green_file)
		with open('preloaded_models/blue_model.pkl', 'rb') as blue_file:
			blue_model = pickle.load(blue_file)
	else:
		print('no preloaded models found...')
		# initialize list of training images TODO: optimize loading training images with numpy or multithreading...
		print('loading training images...')
		train_mountain_images = [Image.open(f'mountain_images_training/IMG_19{x}.JPG') for x in range(64, 91)]
		train_mountain_images_data = [np.asarray(image) for image in train_mountain_images]
		train_mountain_images_data_normal = [normalize(image_data) for image_data in train_mountain_images_data]
		print('training images loaded')

		# initialize models for each color channel
		total_iters = 1000
		n_iters = int(total_iters / len(train_mountain_images))  # distribute total iterations over training images
		red_model = LinearRegression(n_iters=n_iters)
		green_model = LinearRegression(n_iters=n_iters)
		blue_model = LinearRegression(n_iters=n_iters)

		# train models in parallel FIXME: this uses a lot of memory...
		# n_jobs = 2
		# train_image_data_normal_chunks = np.array_split(train_mountain_images_data_normal, n_jobs)
		# Parallel(n_jobs=n_jobs)(
		# 	delayed(batch_train)(
		# 		split_image_data(normal_image_data_chunk),
		# 		red_model.fit,
		# 		green_model.fit,
		# 		blue_model.fit
		# 	) for normal_image_data_chunk in train_mountain_images_data_normal
		# )

		# train models from training data
		print('begin training models...')
		[batch_train(
			split_image_data(normal_train_data),
			red_model.fit,
			green_model.fit,
			blue_model.fit
		) for normal_train_data in train_mountain_images_data_normal]
		print('training complete')

		with open('preloaded_models/red_model.pkl', 'wb') as red_file:
			pickle.dump(red_model, red_file)
		with open('preloaded_models/green_model.pkl', 'wb') as green_file:
			pickle.dump(green_model, green_file)
		with open('preloaded_models/blue_model.pkl', 'wb') as blue_file:
			pickle.dump(blue_model, blue_file)

	new_image_height = test_mountain_image_data.shape[0]
	new_image_width = test_mountain_image_data.shape[1]

	new_red = red_model.predict(test_data_split['green_blue'])
	new_green = green_model.predict(test_data_split['red_blue'])
	new_blue = blue_model.predict(test_data_split['red_green'])
	new_rgb_flat = denormalize(np.stack((new_red, new_green, new_blue), axis=-1))

	# print(test_mountain_image_data.shape)
	# print(new_rgb_flat.shape)
	new_rgb_shaped = new_rgb_flat.reshape((new_image_height, new_image_width, 3))
	# print(new_rgb_shaped.shape)

	output_image = Image.fromarray(new_rgb_shaped)
	output_image.save('output_images/test2.jpg')


def process_2():
	"""
	second process: train models for tiles of stitched-image, test models on those tiles
	:return: void
	"""
	image = Image.open('sedona_images_4-22/solophotos/IMG_2084.JPG')
	image_data = np.asarray(image)
	image_height = image_data.shape[0]
	image_width = image_data.shape[1]
	image_struct = ImageStruct('img_2084.jpg', image_data)
	[model.fit() for model in image_struct.color_models.values()]
	new_red = image_struct.color_models['red'].predict()
	new_green = image_struct.color_models['green'].predict()
	new_blue = image_struct.color_models['blue'].predict()
	new_rgb_flat = denormalize(np.stack((new_red, new_green, new_blue), axis=-1))
	new_rgb_shaped = new_rgb_flat.reshape((image_height, image_width, 3))
	new_image = Image.fromarray(new_rgb_shaped)
	new_image.save('sedona_images_4-22/_outputs/gorspy_p2_img_2084.jpg')


def process_3():
	"""
	third process: train models on source images and test them on the same images, then stitch them into output
	:return: void
	"""
	# load images into dictionary
	original_images = load_image_directory('mountain_images_training')
	# TODO: finish this process


def load_image_directory(directory):
	"""
	loads a numpy image arrays of .jpg files
	:param directory: absolute or relative path of directory containing .jpg images to process
	:return: list of numpy image arrays for each .jpg in the directory
	"""
	image_files = os.listdir(directory)
	images = [Image.open(f'{directory}/{file_name}') for file_name in image_files]  # TODO: add filter to select only jpgs
	images_data = {file_name: np.asarray(image) for (file_name, image) in zip(image_files, images)}  # TODO: figure out how to fix this warning...
	return images_data


def normalize(image_data):
	"""
	normalizes an image array
	:param image_data: numpy image array of values between 0 and 255
	:return: normal_data numpy array of values between 0 and 1
	"""
	normal_data = image_data / 255
	return normal_data


def denormalize(normal_data):
	"""
	converts normal data into an image
	:param normal_data: numpy array of values between 0 and 1
	:return: image_data numpy array of values between 0 and 255
	"""
	image_data = normal_data * 255
	return image_data.astype(np.uint8)


def split_image_data(image_data):
	"""
	splits image data into tuple of arrays to work with the linear regression model
	:param image_data: 2D numpy array of pixels
	:return: formatted dictionary with members formatted for LinearRegression model
	"""
	n_samples = image_data.shape[0] * image_data.shape[1]
	n_features = 3
	flat_pixels = image_data.reshape(n_samples, n_features)

	# isolate red, green, and blue channels
	red, green, blue = np.hsplit(flat_pixels, n_features)

	# flatten the arrays
	red = red.flatten()
	green = green.flatten()
	blue = blue.flatten()

	# combine them into pairs
	green_blue = np.stack((green, blue), axis=-1)
	red_blue = np.stack((red, blue), axis=-1)
	red_green = np.stack((red, green), axis=-1)

	return {
		'red': red,
		'green_blue': green_blue,
		'green': green,
		'red_blue': red_blue,
		'blue': blue,
		'red_green': red_green,
	}


def batch_train(normal_image_split_dict, fit_red, fit_green, fit_blue):
	"""
	function used in comprehension to train each color model
	:param normal_image_split_dict: split formatted dictionary of the images color information
	:param fit_red: reference to red model fit method
	:param fit_green: reference to green model fit method
	:param fit_blue: reference to blue model fit method
	:return: void
	"""
	n_jobs = 4
	# print('training red')
	fit_red(
		training_input=normal_image_split_dict['green_blue'],
		training_output=normal_image_split_dict['red'],
		n_jobs=n_jobs
	)
	# print('training green')
	fit_green(
		training_input=normal_image_split_dict['red_blue'],
		training_output=normal_image_split_dict['green'],
		n_jobs=n_jobs
	)
	# print('training blue')
	fit_blue(
		training_input=normal_image_split_dict['red_green'],
		training_output=normal_image_split_dict['blue'],
		n_jobs=n_jobs
	)

	print('batch complete')


class LinearRegression:
	def __init__(self, learning_rate=0.01, n_iters=1000):
		self.learning_rate = learning_rate
		self.n_iters = n_iters
		self.weights = None
		self.bias = None

	def fit(self, training_input, training_output, n_jobs=-1):
		n_samples, n_features = training_input.shape
		# print(n_samples, n_features)
		if self.weights is None and self.bias is None:
			self.weights = np.zeros(n_features)
			self.bias = 0

		for _ in range(self.n_iters):
			predicted_output = np.dot(training_input, self.weights) + self.bias

			dw = (1 / n_samples) * np.dot(training_input.T, (predicted_output - training_output))
			db = (1 / n_samples) * np.sum(predicted_output - training_output)

			self.weights -= self.learning_rate * dw
			self.bias -= self.learning_rate * db

	# FIXME: correctly parallelize this...
	# def update_weights(input_chunk, output_chunk):
	# 	predicted_output = np.dot(input_chunk, self.weights) + self.bias
	#
	# 	dw = (1 / n_samples) * np.dot(input_chunk.T, (predicted_output - output_chunk))
	# 	db = (1 / n_samples) * np.sum(predicted_output - output_chunk)
	#
	# 	self.weights -= self.learning_rate * dw
	# 	self.bias -= self.learning_rate * db
	#
	# in_chunks = np.array_split(training_input, n_jobs)
	# out_chunks = np.array_split(training_output, n_jobs)
	#
	# Parallel(n_jobs=n_jobs)(
	# 	delayed(update_weights)(in_chunk, out_chunk) for in_chunk, out_chunk in zip(in_chunks, out_chunks)
	# )

	def predict(self, testing_input):
		return np.dot(testing_input, self.weights) + self.bias


class PixelPredictorLR:
	# TODO: add ability to load predictor from json
	def __init__(self, train_input, train_output, learning_rate=0.01, n_iters=1000):
		self.train_input = train_input
		self.train_output = train_output
		self.learning_rate = learning_rate
		self.n_iters = n_iters
		self.weights = None
		self.bias = None

	def fit(self):
		n_samples, n_features = self.train_input.shape
		self.weights = np.zeros(n_features)
		self.bias = 0

		for _ in range(self.n_iters):
			predicted_output = np.dot(self.train_input, self.weights) + self.bias

			dw = (1 / n_samples) * np.dot(self.train_input.T, (predicted_output - self.train_output))
			db = (1 / n_samples) * np.sum(predicted_output - self.train_output)

			self.weights -= self.learning_rate * dw
			self.bias -= self.learning_rate * db

	def predict(self, test_input=None):
		return np.dot(self.train_input, self.weights) + self.bias


class ImageStruct:
	def __init__(self, file_name, image_data):
		self.file_name = file_name

		normal_split_data = split_image_data(normalize(image_data))
		self.color_models = {
			'red': PixelPredictorLR(normal_split_data['green_blue'], normal_split_data['red'], n_iters=100),
			'green': PixelPredictorLR(normal_split_data['red_blue'], normal_split_data['green'], n_iters=100),
			'blue': PixelPredictorLR(normal_split_data['red_green'], normal_split_data['blue'], n_iters=100)
		}


if __name__ == '__main__':
	main()

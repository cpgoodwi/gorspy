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
	process_3()


def process_2(input_directory, output_directory, filename, image_data=None):
	"""
	second process: train models for tiles of image, test models on those tiles
	:return: none
	"""
	# open image and convert it to numpy array, saving height and width
	if image_data is None:
		image = Image.open(f'{input_directory}/{filename}')
		image_data = np.asarray(image)
		image_height = image_data.shape[0]
		image_width = image_data.shape[1]

	# turn the image into tiles
	n_tiles_y = 8
	n_tiles_x = 8
	tiles = np.array_split(image_data, n_tiles_y, axis=0)
	tiles = np.asarray([np.array_split(tile, n_tiles_x, axis=1) for tile in tiles])
	tile_height = tiles.shape[2]
	tile_width = tiles.shape[3]
	# print(np.asarray(tiles).shape)

	# run models on the tiles and paste the image together
	tile_structs = [
		ImageStruct(f'{filename}_{row}_{col}', tiles[row][col])
		for row in range(n_tiles_y)
		for col in range(n_tiles_x)
	]
	# fit each model on each tile TODO: parallelize fitting the models for each tile
	[model.fit() for tile_struct in tile_structs for model in tile_struct.color_models.values()]
	predicted_output = np.asarray([predict_struct(tile_struct) for tile_struct in tile_structs])
	shaped_tiles_out = predicted_output.reshape((n_tiles_y, n_tiles_x, tile_height, tile_width, 3))

	# stitch tiles back together
	rows = [np.concatenate(row, axis=1) for row in shaped_tiles_out]
	reshaped_data = np.concatenate(rows, axis=0)

	# output the new image
	new_image = Image.fromarray(reshaped_data)
	new_image.save(f'{output_directory}/gorspy_p2_tiles_{filename}')


def process_3():
	"""
	third process: train models on source images and test them on the same images, then stitch them into output
	:return: none
	"""
	input_directory = 'phoenixmountainpreserve_images/gigapixel1'
	output_directory = 'phoenixmountainpreserve_images/_outputs/gp1_tiled/8x8'

	# load images into dictionary
	original_images = load_image_directory(input_directory)
	# process each image of the dictionary
	[
		process_2(input_directory, output_directory, filename, image_data)
		for filename, image_data in original_images.items()
	]


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


def predict_struct(image_struct):
	new_red = image_struct.color_models['red'].predict()
	new_green = image_struct.color_models['green'].predict()
	new_blue = image_struct.color_models['blue'].predict()
	new_rgb_flat = denormalize(np.stack((new_red, new_green, new_blue), axis=-1))
	new_rgb_shaped = new_rgb_flat.reshape((image_struct.height, image_struct.width, 3))
	return new_rgb_shaped


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
		self.height = image_data.shape[0]
		self.width = image_data.shape[1]

		normal_split_data = split_image_data(normalize(image_data))
		self.color_models = {
			'red': PixelPredictorLR(normal_split_data['green_blue'], normal_split_data['red'], n_iters=100),
			'green': PixelPredictorLR(normal_split_data['red_blue'], normal_split_data['green'], n_iters=100),
			'blue': PixelPredictorLR(normal_split_data['red_green'], normal_split_data['blue'], n_iters=100)
		}


if __name__ == '__main__':
	main()

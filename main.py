from PIL import Image
import numpy as np


def main():
	test_mountain_image = Image.open('mountain_images_testing/phoenix-mountain-preserve-short.jpg')
	test_mountain_image_data = np.asarray(test_mountain_image)
	test_mountain_image_data_normal = normalize(test_mountain_image_data)

	# train_mountain_images = [Image.open(f'mountain_images_training/IMG_19{x}.JPG') for x in range(64, 91)]
	# train_mountain_images_data = [np.asarray(image) for image in train_mountain_images]
	# train_mountain_images_data_normal = [normalize(image_data) for image_data in train_mountain_images_data]

	# print(test_mountain_image_data[1])
	test_green_blue = test_mountain_image_data[:, :, 1:]  # blue and green values
	test_red = test_mountain_image_data[:, :, 0]  # red info of image
	print(split_image_data(test_mountain_image_data))

	# print(test_mountain_image_data[:, :, 0])
	# print(test_mountain_image_data[1, :2])

	# red_model = LinearRegression()

	# output_image = Image.fromarray(mountain_images_data[1])
	# output_image.save('output_images/test.png')  # this works!


def normalize(image_data):
	"""
	Normalizes an image array
	:param image_data: numpy image array of values between 0 and 255
	:return: normal_data numpy array of values between 0 and 1
	"""
	normal_data = image_data / 255
	return normal_data


def denormalize(normal_data):
	"""
	Converts normal data into an image
	:param normal_data: numpy array of values between 0 and 1
	:return: image_data numpy array of values between 0 and 255
	"""
	image_data = normal_data * 255
	return image_data.astype(int)


def split_image_data(image_data):
	"""
	Splits image data into tuple of arrays to work with the linear regression model
	:param image_data: 2D numpy array of pixels
	:return: reshaped numpy array to flatten into a 1D array of pixels
	"""
	n_samples = image_data.shape[0] * image_data.shape[1]
	n_features = 3
	return image_data.reshape(n_samples, n_features)


class LinearRegression:
	def __init__(self, learning_rate=0.01, n_iters=1000):
		self.learning_rate = learning_rate
		self.n_iters = n_iters
		self.weights = None
		self.bias = None

	def fit(self, training_input, training_output):
		n_samples, n_features = training_input.shape
		self.weights = np.zeros(n_features)
		self.bias = 0

		for _ in range(self.n_iters):
			predicted_output = np.dot(training_input, self.weights) + self.bias

			dw = (1 / n_samples) * np.dot(training_input.T, (predicted_output - training_output))
			db = (1 / n_samples) * np.sum(predicted_output - training_output)

			self.weights -= self.learning_rate * dw
			self.bias -= self.learning_rate * db

	def predict(self, testing_input):
		return np.dot(testing_input, self.weights) + self.bias


if __name__ == "__main__":
	main()

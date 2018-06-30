// Main JavaScript for TensorFlow.js Variational Autoencoder

// Define settings for variational autoencoder
// Size of input and output images in pixels (width and height)
const imageSize = 16;
// Number of images to use when training the neural network
const numTrainingImages = 46;
// Number of thumbnail canvases to display randomly generated images on
const numCanvases = 2;

// Automatically generated settings and parameters
// Volume of image data, calculated by squaring imageSize to find the area of the image (total number of pixels) and multiplying by three for each color channel (RGB)
const imageVolume = (imageSize ** 2) * 3;
// Value to multiply pixels by to scale values from 0 - 255 to 0 - 1
const pixelMul = tf.scalar(1 / 255);

// Get information for main canvas elements
const canvas = {
	// Get information for input canvas to display randomly selected input image for the autoencoder
	"input": document.getElementById("inputCanvas"),
	// Get information for output canvas to display autoencoded representation of the original input image
	"output": document.getElementById("outputCanvas"),
	// Get information for "reconstruction" canvas that displays the output from a user-defined set of latent-space variables
	"reconstruction": document.getElementById("reconstructionCanvas")
}
// Get context for main canvas elements
const context = {
	// Get context for input canvas
	"input": canvas.input.getContext("2d"),
	// Get context for output canvas
	"output": canvas.output.getContext("2d"),
	// Get context for reconstruction canvas
	"reconstruction": canvas.reconstruction.getContext("2d")
}

// Set canvas dimensions to match specified image dimensions
// Input canvas
canvas.input.width = imageSize;
canvas.input.height = imageSize;
// Output canvas
canvas.output.width = imageSize;
canvas.output.height = imageSize;
// Reconstruction canvas
canvas.reconstruction.width = imageSize;
canvas.reconstruction.height = imageSize;

// Define encoder network with the high-level TensorFlow.js layers system
// This network takes a high-dimensional input image and reduces it to a low-dimensional "latent-space" representation
// Define encoder network layers
const encoder = {
	// Input layer with the same number of units as the volume of the input image
	input: tf.input({shape: imageVolume}),
	// Hidden layers
	hidden: [
		// First hidden layer - dense layer with 500 units and a relu activation function
		tf.layers.dense({units: 500, activation: "relu"}),
		// Second hidden layer - dense layer with 300 units and a relu activation function
		tf.layers.dense({units: 300, activation: "relu"}),
		// Third hidden layer - dense layer with 100 units and a relu activation function
		tf.layers.dense({units: 100, activation: "relu"})
	]
};
// Define data flow through encoder model layers
// Output layer is a dense layer with 5 units that is calculated by applying the third ([2]) hidden layer
encoder.output = tf.layers.dense({units: 5}).apply(
	// Third hidden layer is calculated by applying the second ([1]) hidden layer
	encoder.hidden[2].apply(
		// Third hidden layer is calculated by applying the first ([0]) hidden layer
		encoder.hidden[1].apply(
			// First hidden layer is calculated by applying the input
			encoder.hidden[0].apply(
				// Encoder network input
				encoder.input
			)
		)
	)
);

// Define decoder network
// This network takes a low-dimensional "latent-space" representation of the input image (created by the encoder network) and creates a high-dimensional output image (meant to match the original input image)
// Define decoder network layers
const decoder = {
	// Input layer with the same number of units as the output of the encoder network (the number of latent variables)
	input: tf.input({shape: 5}),
	// Hidden layers
	hidden: [
		// First hidden layer - dense layer with 100 units and a relu activation function
		tf.layers.dense({units: 100, activation: "relu"}),
		// Second hidden layer - dense layer with 300 units and a relu activation function
		tf.layers.dense({units: 300, activation: "relu"}),
		// Third hidden layer - dense layer with 500 units and a relu activation function
		tf.layers.dense({units: 500, activation: "relu"})
	]
};
// Define data flow through decoder model layers
// Output layer is a dense layer with the same number of units as the input image/data that is calculated by applying the third ([2]) hidden layer
decoder.output = tf.layers.dense({units: imageVolume}).apply(
	// Third hidden layer is calculated by applying the second ([1]) hidden layer
	decoder.hidden[2].apply(
		// Third hidden layer is calculated by applying the first ([0]) hidden layer
		decoder.hidden[1].apply(
			// First hidden layer is calculated by applying the input
			decoder.hidden[0].apply(
				// Decoder network input
				decoder.input
			)
		)
	)
);

// Create a new TensorFlow.js model to act as the encoder network in the autoencoder
encoder.model = tf.model(
	{
		// Set inputs to predefined encoder network input layer
		inputs: encoder.input,
		// Set outputs to predefined encoder network outputs layer
		outputs: encoder.output
	}
);
// Create a new model to act as the decoder network in the autoencoder
decoder.model = tf.model(
	{
		// Set inputs to predefined decoder network input layer
		inputs: decoder.input,
		// Set outputs to predefined decoder network outputs layer
		outputs: decoder.output
	}
);

// Neural network training/optimization
// Define loss function for neural network training: Mean squared error
loss = (input, output) => input.sub(output).square().mean();
// Learning rate for optimization algorithm
const learningRate = 0.001;
// Optimization function for training neural networks
optimizer = tf.train.adam(learningRate);

// Loss calculation function for variational autoencoder neural network
const calculateLoss =
// Wrap loss calculation function in a tf.tidy so that intermediate tensors are disposed of when the calculation is finished
() => tf.tidy(
	// Calculate loss
	() => {
		return tf.add(
			// Evaluate the loss function given the output of the autoencoder network and the actual image
			loss(
				// Pass the input data through the autoencoder
				decoder.model.predict(encoder.model.predict(trainingData.tensor.mul(pixelMul))),
				trainingData.tensor
			),
			// Evaluate the divergence of the network from a normal distribution (KL divergence)
			loss(
				// Create a latent representation of the input data with the encoder network
				encoder.model.predict(trainingData.tensor.mul(pixelMul)),
				// Generate a tensor of random normal values
				tf.randomNormal([5])
			)
		);
	}
);

// Create object to store training data in image, pixel, and tensor format
const trainingData = {
	// Store training data image elements
	"images": [],
	// Store training data as raw arrays of pixel data
	"pixels": []
}
// Add training data to trainingData.images array as an HTML image element
// Loop through each training image
for (var i = 0; i < numTrainingImages; i ++) {
	// Create a new HTML image element with the specified dimensions and set current array index to this element (array.push does not work here)
	trainingData.images[i] = new Image(imageSize, imageSize);
}

// Wait for images (training data) to load before continuing
trainingData.images[trainingData.images.length - 1].onload = function () {
	// Create training data from pixels of image elements
	// Create a new variable to store the data
	var pixels;
	// Loop through each training image
	for (var i = 0; i < numTrainingImages; i ++) {
		// Create a tensor with 3 (RGB) color channels from the image element
		pixels = tf.fromPixels(trainingData.images[i], 3);
		// Resize image to the specified dimensions with resizeBilinear()
		pixels = tf.image.resizeBilinear(pixels, [imageSize, imageSize]);
		// Get the values array from the pixels tensor
		pixels = pixels.dataSync();
		// Add new array to trainingData.pixels to store the pixel values for the image
		trainingData.pixels.push([]);
		// Loop through each value in the pixels array
		// The whole pixels array is not pushed on at once because the array format will be incompatible
		pixels.forEach(
			// Add color value to the corresponding image's trainingData.pixels array
			(element) => trainingData.pixels[i].push(element)
		);
	}
	// Create a tensor from the pixel values of the training data and store it in trainingData.tensor
	trainingData.tensor = tf.tensor(trainingData.pixels);

	// Pick a random image from the training data to test the network on
	var index = Math.floor(Math.random() * trainingData.pixels.length);

	// Create image tensor from input image pixel data
	const input = tf.tensor(trainingData.pixels[index], [imageSize, imageSize, 3]);
	// Set input image tensor dtype to "int32"
	input.dtype = "int32";
	// Display input imageon the input canvas, then dispose of the input tensor
	tf.toPixels(input, canvas.input).then(() => input.dispose());

	// Function for limiting the pixel values of output images to a 0 - 255 range (outdated, replaced with clipByValue)
	function limitPixels(pixels) {
		// Get pixel values as an array from input tensor
		var values = pixels.dataSync();
		// Loop through each value
		for (var i = 0; i < values.length; i ++) {
			// Check if value is less than 0
			if (values[i] < 0) {
				// Set the value to 0
				values[i] = 0;
			}
			// Check if value is greater than 255
			else if (values[i] > 255) {
				// Set the value to 255
				values[i] = 255;
			}
		}
		// Return the data as an image-formatted tensor with dtype of "int32"
		return tf.tensor(values, [imageSize, imageSize, 3], "int32");
	}

	// Create thumbnail canvases to render randomly generated images
	// Create array to store thumbnail canvas elements
	const canvases = [];
	// Calculate minumum value of latent variables - this will be used to generate latent variables for new images
	const min = encoder.model.predict(trainingData.tensor).min().dataSync()[0];
	// Calculate maximum value of latent variables
	const max = encoder.model.predict(trainingData.tensor).max().dataSync()[0];
	// Create a specified number of new canvases
	for (var i = 0; i < numCanvases; i ++) {
		// Create a new HTML canvas element
		var element = document.createElement("canvas");
		// Add a corresponding id property to the canvas element
		element.id = "canvas-" + i;
		// Set the "thumbnail" CSS class for the new canvas
		element.className = "thumbnail";
		// Add the canvas element to the page body
		document.body.appendChild(element);
		// Add this canvas element to the canvases array
		canvases.push({
			// Select the canvas element by id and add it to the array
			"canvas": document.getElementById("canvas-" + i),
			// Add randomly generated latent space variables for this canvas
			// "variables": tf.randomNormal([1, 5])
			"variables": tf.randomUniform([1, 5], min, max)
		});
		// Add rendering context object for the canvas
		canvases[i].context = canvases[i].canvas.getContext("2d");
	}

	// Define training function for variational autoencoder neural network - this will be executed iteratively
	function train() {
		// Print TensorFlow.js memory information to console, including the number of tensors stored in memory (for debugging purposes)
		console.log(tf.memory());
		// Use tidy here
		// Print current neural network loss to console
		// Calculate loss value and store it in a constant
		const printLoss = calculateLoss();
		// Print loss to console
		printLoss.print();
		// Dispose of loss value
		printLoss.dispose();

		// Minimize the error/cost calculated by the loss calculation funcion using the optimization function
		optimizer.minimize(calculateLoss);

		// All this is just display code
		// Calculate autoencoder output from original image
		const output =
		// Wrap output calculation in a tf.tidy() to remove intermediate tensors after the calculation is complete
		tf.tidy(
			() => {
				// Decode the low-dimensional representation of the input data created by the encoder
				return decoder.model.predict(
					// Create an encoded (low-dimensional) representation of the input data
					encoder.model.predict(
						// Create a tensor from the array of pixel values for the randomly selected input image
						tf.tensor(
							[trainingData.pixels[index]]
						)
						// Multiply the input data tensor by the pixel coefficient
						.mul(pixelMul)
					)
				)
				// Reduce output values from ~ 0 - 255 to ~ 0 - 1
				.mul(pixelMul)
				// Clip pixel values to a 0 - 1 (float32) range
				.clipByValue(0, 1)
				// Reshape the output tensor into an image format (w * l * 3)
				.reshape(
					[imageSize, imageSize, 3]
				)
			}
		);

		// Display the output tensor on the output canvas, then dispose the tensor
		tf.toPixels(output, canvas.output).then(() => output.dispose());

		// Display randomly generated images on thumbnail canvases
		// Loop through each canvas
		for (var i = 0; i < canvases.length; i ++) {
			const output_ =
			// Wrap output calculation in a tf.tidy() to remove intermediate tensors after the calculation is complete
			tf.tidy(
				() => {
					// Generate output given randomly generated latent variables
					return decoder.model.predict(canvases[i].variables)
					// Reduce output values from ~ 0 - 255 to ~ 0 - 1
					.mul(pixelMul)
					// Clip pixel values to a 0 - 1 (float32) range
					.clipByValue(0, 1)
					// Reshape the output tensor into an image format (w * l * 3)
					.reshape(
						[imageSize, imageSize, 3]
					);
				}
			);
			// Display the output tensor on the output canvas, then dispose the tensor
			tf.toPixels(output_, canvases[i].canvas).then(() => output_.dispose());
		}
	}
	// Set an interval of 100 milliseconds to repeat the train() function
	var interval = window.setInterval(train, 100);
}
// Load source paths for training data images (this must be done after the image elements are created and the onload function is defined)
// Loop through each image element
for (var i = 0; i < numTrainingImages; i ++) {
	// Set the corresponding source for the image
	trainingData.images[i].src = "./Training Data/Characters/" + (i + 1) + ".png";
}

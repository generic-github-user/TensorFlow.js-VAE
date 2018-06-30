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

canvas = {
	"input": document.getElementById("inputCanvas"),
	"output": document.getElementById("outputCanvas"),
	"reconstruction": document.getElementById("reconstructionCanvas")
}
context = {
	"input": canvas.input.getContext("2d"),
	"output": canvas.output.getContext("2d"),
	"reconstruction": canvas.reconstruction.getContext("2d")
}

canvas.input.width = imageSize;
canvas.input.height = imageSize;
canvas.output.width = imageSize;
canvas.output.height = imageSize;
canvas.reconstruction.width = imageSize;
canvas.reconstruction.height = imageSize;

const encoder = {
	input: tf.input({shape: imageVolume}),
	hidden: [
		tf.layers.dense({units: 500, activation: "relu"}),
		tf.layers.dense({units: 300, activation: "relu"}),
		tf.layers.dense({units: 100, activation: "relu"})
	]
};
encoder.output = tf.layers.dense({units: 5}).apply(
	encoder.hidden[2].apply(
		encoder.hidden[1].apply(
			encoder.hidden[0].apply(
				encoder.input
			)
		)
	)
);

const decoder = {
	input: tf.input({shape: 5}),
	hidden: [
		tf.layers.dense({units: 100, activation: "relu"}),
		tf.layers.dense({units: 300, activation: "relu"}),
		tf.layers.dense({units: 500, activation: "relu"})
	]
};
decoder.output = tf.layers.dense({units: imageVolume}).apply(
	decoder.hidden[2].apply(
		decoder.hidden[1].apply(
			decoder.hidden[0].apply(
				decoder.input
			)
		)
	)
);

encoder.model = tf.model(
	{
		inputs: encoder.input,
		outputs: encoder.output
	}
);
decoder.model = tf.model(
	{
		inputs: decoder.input,
		outputs: decoder.output
	}
);

//
// Define loss function for neural network training: Mean squared error
loss = (input, output) => input.sub(output).square().mean();
// Learning rate for optimization algorithm
const learningRate = 0.001;
// Optimization function for training neural networks
optimizer = tf.train.adam(learningRate);

// Loss calculation function for variational autoencoder neural network
const calculateLoss =
() => tf.tidy(
	() => {
		return tf.add(
			loss(
				decoder.model.predict(encoder.model.predict(trainingData.tensor.mul(pixelMul))),
				trainingData.tensor
			),
			loss(
				encoder.model.predict(trainingData.tensor.mul(pixelMul)),
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

trainingData.images[trainingData.images.length - 1].onload = function () {
	var pixels;
	for (var i = 0; i < numTrainingImages; i ++) {
		pixels = tf.fromPixels(trainingData.images[i], 3);
		pixels = tf.image.resizeBilinear(pixels, [imageSize, imageSize]);
		pixels = pixels.dataSync();
		trainingData.pixels.push([]);
		pixels.forEach(
			(element) => trainingData.pixels[i].push(element)
		);
	}

	trainingData.tensor = tf.tensor(trainingData.pixels);

	var index = Math.floor(Math.random() * trainingData.pixels.length);

	const input = tf.tensor(trainingData.pixels[index], [imageSize, imageSize, 3]);
	input.dtype = "int32";
	tf.toPixels(input, canvas.input).then(() => input.dispose());

	function limitPixels(pixels) {
		var values = pixels.dataSync();
		for (var i = 0; i < values.length; i ++) {
			if (values[i] < 0) {
				values[i] = 0;
			}
			else if (values[i] > 255) {
				values[i] = 255;
			}
		}
		return tf.tensor(values, [imageSize, imageSize, 3], "int32");
	}

	// Create thumbnail canvases to render randomly generated images
	// Create array to store thumbnail canvas elements
	const canvases = [];
	const min = encoder.model.predict(trainingData.tensor).min().dataSync()[0];
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
			"variables": tf.randomNormal([1, 5], -10, 10)
		});
		// Add rendering context object for the canvas
		canvases[i].context = canvases[i].canvas.getContext("2d");
	}

	function train() {
		console.log(tf.memory());
		// Use tidy
		const printLoss = calculateLoss();
		printLoss.print();
		printLoss.dispose();

		optimizer.minimize(calculateLoss);
		// All this is just display code
		const output =
		tf.tidy(
			() => {
				// Decode the low-dimensional representation of the input data created by the encoder
				return decoder.model.predict(
					// Create an encoded (low-dimensional) representation of the input data
					encoder.model.predict(
						tf.tensor(
							[trainingData.pixels[index]]
						)
						// Multiply the input data tensor by the pixel coefficient
						.mul(pixelMul)
					)
				)
				.mul(pixelMul)
				.clipByValue(0, 1)
				.reshape(
					[imageSize, imageSize, 3]
				)
			}
		);

		tf.toPixels(output, canvas.output).then(() => output.dispose());

		for (var i = 0; i < canvases.length; i ++) {
			const output_ =
			tf.tidy(
				() => {
					return decoder.model.predict(canvases[i].variables)
					.mul(pixelMul)
					.clipByValue(0, 1)
					.reshape(
						[imageSize, imageSize, 3]
					);
				}
			);
			tf.toPixels(output_, canvases[i].canvas).then(() => output_.dispose());
		}
	}
	var interval = window.setInterval(train, 100);
}
for (var i = 0; i < numTrainingImages; i ++) {
	trainingData.images[i].src = "./Training Data/Characters/" + (i + 1) + ".png";
}

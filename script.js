const imageSize = 16;
const trainingImages = 46;

const imageVolume = (imageSize ** 2) * 3;
const pixelMul = tf.scalar(1 / 255);
// const layerSizes = [1, 2, 4, 8, 4, 2, 1];

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
//pixel value floats or ints
canvas.input.width = imageSize;
canvas.input.height = imageSize;
canvas.output.width = imageSize;
canvas.output.height = imageSize;
canvas.reconstruction.width = imageSize;
canvas.reconstruction.height = imageSize;
//get random value limits from hidden layer
const canvases = [];
for (var i = 0; i < 2; i ++) {
	var element = document.createElement("canvas");
	element.id = "canvas-" + i;
	element.className = "thumbnail";
	document.body.appendChild(element);
	// document.body.innerHTML += "<canvas id='canvas-" + i + "' width='256' height='256'></canvas>";
	canvases.push({
		"canvas": document.getElementById("canvas-" + i),
		"parameters": tf.randomNormal([1, 5], -10, 10)
	});
	canvases[i].context = canvases[i].canvas.getContext("2d");
}
// best layer size ratios
//reduce weight values
//wait for optimizer to escape local minimum
//crop images
//yeah, I shouldn't start any new projects
//trainingData.tensor.matMul(encoder.weights[0]).add(encoder.biases[0].matMul(encoder.weights[0])).tanh().print()

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
//this may not have been happening within the *model*

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

//has died
//1-line JSON
//over-tanh-ing
//lots of tweaking
//just add biases - let the optimizer figure it out
//hidden layer activation function
//flowcharts
//add the biases before applying the activation function

//
// Define loss function for neural network training: Mean squared error
loss = (input, output) => input.sub(output).square().mean();
// Learning rate for optimization algorithm
// const learningRate = 0.000001;
const learningRate = 0.001;
// Optimization function for training neural networks
// optimizer = tf.train.momentum(learningRate, 0.9);
optimizer = tf.train.adam(learningRate);
// optimizer = tf.train.momentum(learningRate, 0.1);
// optimizer = tf.train.rmsprop(learningRate);
//just look everything up
//shuffle training data
//optimizer?
//4-dimensional convolutions
//does sgd work?
//local minimum for a while?

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
//pixelmul
//variable names
const trainingData = {
	"images": [],
	"pixels": []
}
for (var i = 0; i < trainingImages; i ++) {
	trainingData.images[i] = new Image(imageSize, imageSize);
}

trainingData.images[trainingData.images.length - 1].onload = function () {
	var pixels;
	for (var i = 0; i < trainingImages; i ++) {
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
	tf.toPixels(input, canvas.input);
	//input.dispose();

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

	//tidy
	const canvases = [];
	const min = encoder.model.predict(trainingData.tensor).min().dataSync()[0];
	const max = encoder.model.predict(trainingData.tensor).max().dataSync()[0];
	for (var i = 0; i < 4; i ++) {
		var element = document.createElement("canvas");
		element.id = "canvas-" + i;
		element.className = "thumbnail";
		document.body.appendChild(element);
		// document.body.innerHTML += "<canvas id='canvas-" + i + "' width='256' height='256'></canvas>";
		canvases.push({
			"canvas": document.getElementById("canvas-" + i),
			"parameters": tf.randomUniform([1, 5], min, max)
		});
		canvases[i].context = canvases[i].canvas.getContext("2d");
	}

	function train() {
		// for (var i = 0; i < 10; i ++) {
		console.log("1-" + tf.memory().numTensors);
		const printLoss = calculateLoss();
		printLoss.print();
		printLoss.dispose();
		//use tidy ^
		console.log("2-" + tf.memory().numTensors);
		optimizer.minimize(calculateLoss);

		// All this is just display code
		const output =
		tf.tidy(
			() => {
				var output = limitPixels(decoder.model.predict(encoder.model.predict(tf.tensor([trainingData.pixels[index]]).mul(pixelMul))));
				// spread this line out
				output.dtype = "int32";
				return output;
			}
		);
		console.log("3-" + tf.memory().numTensors);
		// output = output.round();
//no errors! because we turned the errors off
		//make this pretty
		// Interestingly, limitPixels(output) cannot be placed inside the tf.tidy() when using ".then(output.dispose());"
		tf.toPixels(output, canvas.output)//.then(output.dispose());

		// tensor is disposed?
		// console.log("4-" + tf.memory().numTensors);
		// for (var i = 0; i < canvases.length; i ++) {
		// 	const output_ =
		// 	tf.tidy(
		// 		() => {
		// 			var output_ = decoder.model.predict(canvases[i].parameters).reshape([imageSize, imageSize, 3]);
		// 			output_.dtype = "int32";
		// 			return output_;
		// 		}
		// 	);
		// 	tf.toPixels(limitPixels(output_), canvases[i].canvas).then(output_.dispose());
		// }
		// console.log("5-" + tf.memory().numTensors);
	}
	var interval = window.setInterval(train, 100);
	//await delays
}
for (var i = 0; i < trainingImages; i ++) {
	// trainingData.images[i].src = "../../Image Sharpening/Feedforward/Training Data/Original/" + (i + 1) + ".jpg";
	trainingData.images[i].src = "./Training Data/Characters/" + (i + 1) + ".png";
}
//reorganize neural nets

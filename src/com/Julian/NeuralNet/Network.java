package com.Julian.NeuralNet;

import java.util.Arrays;

import com.Julian.TrainingSets.TrainSet;

import parser.Attribute;
import parser.Node;
import parser.Parser;
import parser.ParserTools;

public class Network {

	/**
	 * Neural Network 01/16/18
	 * 
	 * @author Julian Abhari
	 */

	// 1. Layer; 2. Neuron
	private double[][] neurons;
	// 1. Layer; 2. Neuron; 3. Weight value connecting neuron to the previous neuron
	private double[][][] weights;
	// 1. Layer; 2. Neuron
	private double[][] bias;
	// The error signal is the sum of the weights connecting the current neuron to
	// the next layer neuron multiplied by the error of the next neuron.
	// If you're trying to calculate the error signal for the last layer however,
	// then it's just the target output minus the actual output.
	private double[][] error_signal;
	// These are the derivatives of all the neuron's outputs.
	private double[][] neurons_derivative;
	// This is an array of how many neurons are in each layer
	public final int[] NETWORK_LAYER_SIZES;
	// Number of neurons in the input layer
	public final int INPUT_SIZE;
	// Number of output neurons in the output layer
	public final int OUTPUT_SIZE;
	// Number of layers
	public final int NETWORK_SIZE;

	// This declares the neurons, weights and biases' array and populates the
	// weights and biases' array
	public Network(int[] NETWORK_LAYER_SIZES) {
		// Making the array = itself
		this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
		// The amount of neurons in the input layer = number in 0 index of array
		this.INPUT_SIZE = NETWORK_LAYER_SIZES[0];
		// The amount of layers
		this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;
		// The amount of output neurons in the output layer
		this.OUTPUT_SIZE = NETWORK_LAYER_SIZES[NETWORK_SIZE - 1];

		// This is declaring how many elements are in the 1st dimension
		// This is the way we can access the neurons layer
		this.neurons = new double[NETWORK_SIZE][];
		// This is how we access the weights
		this.weights = new double[NETWORK_SIZE][][];
		// How we access the neurons' biases
		this.bias = new double[NETWORK_SIZE][];

		// This is declaring how many elements are in the 1st dimension
		this.error_signal = new double[NETWORK_SIZE][];
		// This is declaring how many elements are in the 1st dimension
		this.neurons_derivative = new double[NETWORK_SIZE][];

		for (int i = 0; i < NETWORK_SIZE; i++) {
			// This is declaring how many elements are in the 2nd dimension
			// i.e setting the size of the 2nd dimension
			this.neurons[i] = new double[NETWORK_LAYER_SIZES[i]];
			// ""
			this.error_signal[i] = new double[NETWORK_LAYER_SIZES[i]];
			// ""
			this.neurons_derivative[i] = new double[NETWORK_LAYER_SIZES[i]];

			// This is setting the bias array's layers to random values between 0-1
			this.bias[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i], 0, 1);

			// This is preventing first layer neurons to try to connect to previous neurons
			// that don't exist
			if (i > 0) {
				// Populating the 2nd dimension with arrays (3rd dimension) containing random
				// values between -1-1
				weights[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i], NETWORK_LAYER_SIZES[i - 1], -1, 1);
			}
		}
	}

	// This goes through the entire neural network, populates the neurons with their
	// activations, and calculates the output
	public double[] calculate(double[] input) {
		// If the length of the input is not equal to the input_size that the programmer
		// declared then the function will end
		if (input.length != this.INPUT_SIZE) {
			return null;
		}
		// This is setting the input layer of the "neurons" array to the inputs given to
		// the network
		this.neurons[0] = input;
		// Iterating through the layers of the network
		for (int layer = 1; layer < NETWORK_SIZE; layer += 1) {
			// Iterating through the neruons of the layer
			for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron += 1) {
				// This is the sum value of the weights*activations+bias
				double sum = 0;
				// This is adding the bias neurons of the layer to the sum
				sum += bias[layer][neuron];
				// Iterating through the previousNeurons to get their weights to the next
				// layer's neurons
				for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron += 1) {
					// This is taking the activation of the previous neurons in the previous
					// layer and multiplying it by the weights of the previous neuron
					sum += neurons[layer - 1][prevNeuron] * weights[layer][neuron][prevNeuron];
				}
				// This is setting the activation of each neuron to the result of the sigmoid
				// function, where we give to the weighted sum
				neurons[layer][neuron] = sigmoid(sum);
				// This populates the neuron_derivatives array with the derivatives of every
				// neuron's output
				neurons_derivative[layer][neuron] = (neurons[layer][neuron] * (1 - neurons[layer][neuron]));
			}
		}
		// This returns the activations of the neurons in the output layer
		return neurons[NETWORK_SIZE - 1];
	}

	// This trains the network given a training set and the amount of batches
	// (training examples) neeeded to split the training set
	public void train(TrainSet set, int batchSize, int loops) {
		for (int i = 0; i < loops; i += 1) {
			if (set.INPUT_SIZE != INPUT_SIZE || set.OUTPUT_SIZE != OUTPUT_SIZE)
				return;
			// This extracts batches of the given size (which is typically how many
			// different training examples there are) from the set
			TrainSet batch = set.extractBatch(batchSize);
			// This loops through the batch size (total number of training examples) and
			// trains the network giving a specific learning rate, size of the inputs of the
			// training example, and size of the outputs of the training example
			for (int j = 0; j < batchSize; j += 1) {
				train(batch.getInput(j), batch.getOutput(j), 0.3);
			}
			// System.out.println(meanSquaredError(batch));
		}
	}

	// This function calculates the network's output, calculates the error signals
	// of all the neurons through backpropagation, and adjusts the values of the
	// weights using gradient descent
	private void train(double[] input, double[] target, double learningRate) {
		// If the length of the input is not equal to the input_size that the programmer
		// declared then the function will end
		if (input.length != INPUT_SIZE || target.length != OUTPUT_SIZE) {
			return;
		}
		// This calculates the output of the network when given the inputs
		calculate(input);
		// This calculates the error signals of all the neurons through backpropagation
		// when given the target value
		backpropError(target);
		// This updates the weights and adjusts their values
		updateWeights(learningRate);
	}

	// This returns the mean squared error rate of the network when given a target
	// output vs it's actual output. It calculates how close the neuron's output and
	// target output are. It will return 0 if there is no error
	public double meanSquaredError(double[] input, double[] target) {
		if (input.length != INPUT_SIZE || target.length != OUTPUT_SIZE)
			return 0;
		calculate(input);
		double sum = 0;
		for (int i = 0; i < target.length; i += 1) {
			// This adds the squared error rate to the sum
			sum += (target[i] - neurons[NETWORK_SIZE - 1][i]) * (target[i] - neurons[NETWORK_SIZE - 1][i]);
		}
		return sum / (2.0 * target.length);
	}

	// This returns the mean squared error rate of the network when given a certain
	// training set.
	public double meanSquaredError(TrainSet set) {
		double sum = 0;
		for (int i = 0; i < set.size(); i += 1) {
			sum += meanSquaredError(set.getInput(i), set.getOutput(i));
		}
		return sum / set.size();
	}

	// This function calculates the error signals of all the neurons through
	// backpropagation when given the target outputs
	private void backpropError(double[] target) {
		// This loops through every neuron of the output layer and calculates it's error
		// signal.
		for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[NETWORK_SIZE - 1]; neuron += 1) {
			// This calculates the error signal of the output neurons by getting the
			// neuron's output and subtracting the desired output from it. Then it takes
			// that difference and multiplies it by the derivative of that neuron's output.
			error_signal[NETWORK_SIZE - 1][neuron] = (neurons[NETWORK_SIZE - 1][neuron] - target[neuron])
					* neurons_derivative[NETWORK_SIZE - 1][neuron];
		}
		// This starts at the last hidden layer (the layer right before the output
		// layer) and loops backwards through the layers until it gets to the first
		// hidden layer (the layer right after the input layer).
		for (int layer = NETWORK_SIZE - 2; layer > 0; layer -= 1) {
			// This loops through every neuron of that layer.
			for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron += 1) {
				// This sum value will be used to add up the weights * error signal connecting
				// the current neurons to the next neurons in the layer
				double sum = 0;
				// This loops through every neuron of the next layer
				for (int nextNeuron = 0; nextNeuron < NETWORK_LAYER_SIZES[layer + 1]; nextNeuron += 1) {
					// This adds up the weights connecting the current neuron to every neuron of the
					// next layer * the error_signal of every neuron in the next layer.
					sum += weights[layer + 1][nextNeuron][neuron] * error_signal[layer + 1][nextNeuron];
				}
				// This sets the error signal of the current neuron to the sum * the derivative
				// of the current neuron.
				this.error_signal[layer][neuron] = sum * neurons_derivative[layer][neuron];
			}
		}
	}

	// This method adjusts the values of the weights connecting every neuron and
	// bias neuron using gradient descent
	private void updateWeights(double learningRate) {
		// This loops from the first hidden layer through the rest of the network
		for (int layer = 1; layer < NETWORK_SIZE; layer += 1) {
			// This loops through every neuron in the current layer
			for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron += 1) {
				// This sets the change in the weights (or deltaWeights) to the next step in
				// gradient descent. This calculates the delta weights by multiplying the
				// -learningRate * error signal of the next neurons
				double delta = -learningRate * error_signal[layer][neuron];
				// This adds the delta to the bias neurons whihc changes it's weight value
				// (because it's activation value is just 1
				bias[layer][neuron] += delta;

				// This loops through all the previousNeurons in the whole network and adds the
				// delta * the prevNeuron's output
				for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron += 1) {
					// This gets the weights at the respective indices and adds up the delta * the
					// prevNeuron's output
					weights[layer][neuron][prevNeuron] += delta * neurons[layer - 1][prevNeuron];
				}
			}
		}
	}

	// This is the activation function for all the neurons. It takes a number and
	// squishes it to a rational number between 0.0 and 1.0
	private double sigmoid(double x) {
		return 1.0 / (1 + Math.exp(-x));
	}

	// This takes in a mutationRate and mutates the whole network, going through
	// each layer and each neuron to gently mutate the values of the bias neurons
	// and the weights.
	public Network mutateNetwork(double mutationRate) {
		// This loops from the first hidden layer through the rest of the network
		for (int layer = 1; layer < NETWORK_SIZE; layer += 1) {
			// This loops through every neuron in the current layer
			for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron += 1) {
				bias[layer][neuron] = mutate(bias[layer][neuron], mutationRate, 0, 1);
				// This loops through all the previousNeurons in the whole network
				for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron += 1) {
					weights[layer][neuron][prevNeuron] = mutate(weights[layer][neuron][prevNeuron], mutationRate, -1,
							1);
				}
			}
		}
		return this;
	}

	// This is a mutate funciton which takes in a numberToBeMutated and a
	// mutationRate, if a random number is less than the mutation rate, then a
	// random number between the lowerBound and upperBound will be chosen to
	// offset the given number
	public double mutate(double numberToBeMutated, double mutationRate, double lowerBound, double upperBound) {
		// Random random = new Random();
		double randomNumber = Math.random();
		if (randomNumber < mutationRate) {
			double newNumber = Math.random() * (upperBound - lowerBound) + lowerBound;
			// double offset = (random.nextGaussian() * (upperBound - lowerBound) +
			// lowerBound) * 0.5;
			// double newNumber = numberToBeMutated + offset;
			return newNumber;
		}
		return numberToBeMutated;
	}

	// This loops through the whole network and prints out the weight and bias
	// neuron values for each layer
	public void printNetwork() {
		// This loops from the first hidden layer through the rest of the network
		for (int layer = 1; layer < NETWORK_SIZE; layer += 1) {
			// This loops through every neuron in the current layer
			for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron += 1) {
				System.out.println(
						"Bias neuron at layer: " + layer + ", neuron: " + neuron + " | = " + bias[layer][neuron]);
				// This loops through all the previousNeurons in the whole network
				for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron += 1) {
					System.out.println("Weight at layer: " + layer + ", neuron: " + neuron + ", prevNeuron: "
							+ prevNeuron + " | = " + weights[layer][neuron][prevNeuron]);
				}

			}
		}
	}

	public double[][] getBiasNeurons() {
		return this.bias;
	}

	public double[][][] getWeights() {
		return this.weights;
	}

	public void saveNetwork(String fileName) throws Exception {
		Parser parser = new Parser();
		parser.create(fileName);

		Node root = parser.getContent();
		Node network = new Node("Network");
		Node layers = new Node("Layers");
		network.addAttribute(new Attribute("sizes", Arrays.toString(this.NETWORK_LAYER_SIZES)));
		network.addChild(layers);
		root.addChild(network);

		for (int layer = 1; layer < this.NETWORK_SIZE; layer += 1) {
			Node child = new Node("" + layer);
			layers.addChild(child);
			Node weights = new Node("weights");
			Node biases = new Node("biases");
			child.addChild(weights);
			child.addChild(biases);

			biases.addAttribute("values", Arrays.toString(this.bias[layer]));

			for (int neuron = 0; neuron < this.weights[layer].length; neuron += 1) {
				weights.addAttribute("" + neuron, Arrays.toString(this.weights[layer][neuron]));
			}
		}
		parser.close();
	}

	public static Network loadNetwork(String fileName) throws Exception {
		Parser parser = new Parser();
		parser.load(fileName);

		String sizes = parser.getValue(new String[] { "Network" }, "sizes");
		int[] sizeValues = ParserTools.parseIntArray(sizes);
		Network network = new Network(sizeValues);

		for (int layer = 1; layer < network.NETWORK_SIZE; layer += 1) {
			String biases = parser.getValue(new String[] { "Network", "Layers", new String(layer + ""), "biases" },
					"values");
			double[] bias = ParserTools.parseDoubleArray(biases);
			network.bias[layer] = bias;

			for (int neuron = 0; neuron < network.NETWORK_LAYER_SIZES[layer]; neuron += 1) {
				String currentValue = parser
						.getValue(new String[] { "Network", "Layers", new String(layer + ""), "weights" }, "" + neuron);
				double[] values = ParserTools.parseDoubleArray(currentValue);
				network.weights[layer][neuron] = values;
			}
		}
		parser.close();
		return network;
	}

	public static void main(String[] args) {
		Network neuralNetwork = new Network(new int[] { 2, 3, 3, 1 });

		TrainSet set = new TrainSet(2, 1);

		set.addData(new double[] { 0, 0 }, new double[] { 0 });
		set.addData(new double[] { 0, 1 }, new double[] { 1 });
		set.addData(new double[] { 1, 0 }, new double[] { 1 });
		set.addData(new double[] { 1, 1 }, new double[] { 0 });

		// while the square root of the neuralNetwork's meaanSquaredError rate (when
		// given a set) is > 0.1%, the neural network at that specific example will be
		// trained.
		while (Math.sqrt(neuralNetwork.meanSquaredError(set)) > 0.001) {
			neuralNetwork.train(set, 4, 1);
		}

		for (int trainingExamples = 0; trainingExamples < 4; trainingExamples += 1) {
			System.out.println(Arrays.toString(neuralNetwork.calculate(set.getInput(trainingExamples))));
		}

		try {
			neuralNetwork.saveNetwork("saves/testSave.txt");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
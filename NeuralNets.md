# Neural Networks with MAChineLearning

Neural networks in MAChineLearning currently support:

- multilayer perceptrons of any depth (limited only by memory);
- 4 kinds of activation function:
  - linear,
  - step (0 if output is less than 0.5, 1 if greater),
  - logistic,
  - hyperbolic;
- training through backpropagation with standard gradient descent;
- training both by sample and by epoch;
- load/save of the network status;
- single/double precision (needs recompilation, default is single precision).

Internal code makes heavy use of vDSP and vecLib functions. Compared to the well-known Java library [Neuroph](http://neuroph.sourceforge.net), it is around 20 times faster. It is as fast as it can get on a CPU, but not as fast as it could get with OpenCL on a GPU. An OpenCL version may come in the future.

Following there's a quick tutorial on how to setup and train your neural network.


## Tutorial

### Set up the network

Setting up a network is a matter of one line:

```obj-c
#import <MAChineLearning/MAChineLearning.h>
#import <Accelerate/Accelerate.h>

// Create a perceptron with 3 input lines and 1 output neuron
NeuralNetwork *net= [NeuralNetwork createNetworkWithLayerSizes:@[@3, @1]
outputFunctionType:ActivationFunctionTypeStep];
```

This line creates a single layer perceptron with 3 inputs and 1 output, with step activation function. See the following diagram:

![3-Input Perceptron](3-Input%20Perceptron.png)

When creating a multilayer perceptron, hidden layers always use the logistic activation function, the function of your choice applies only to the output layer.

The import of Accelerate framework is optional and is explained a few paragraphs below.

The network object exposes all you need to control it, namely:

- the input buffer,
- the output buffer,
- the expected output buffer,
- methods to feed forward, back propagate and update weights;
- methods to save the status and create a new network from a saved state.

Buffers are exposed as C arrays for performance reason. They are of type `nnREAL`, which by default is a typedef of `float`. You can redefine this type to `double` in the `NeuralNetworkReal.h` file and then recompile. Just follow the comments.

**Be careful when reading and writing these arrays** since there's no bounds checking, you may easily cause an `EXC_BAD_ACCESS`.

### Loading input

This is how you load your input data:

```obj-c
// Clear the input buffer using vDSP
ML_VDSP_VCLR(net.inputBuffer, 1, net.inputSize);

// Fill appropriate buffer elements
net.inputBuffer[0]= 1.0;
net.inputBuffer[2]= 0.5;
```

To clear the buffer with vDSP you need the import of Accelerate, shown above. Here we use the `ML_VDSP_VCLR` macro, which is appropriately defined for type `nnREAL` in the `NeuralNetworkReal.h` file. Using Accelerate to clear the buffer is advised as it is quicker than a simple loop.

### Computing the output

Once the input buffer is filled, computing the output is simple:

```obj-c
// Compute the output
[net feedForward];

// Log the output
NSLog(@"Output: %.2f", net.outputBuffer[0]);
```

### Training

If the output is not satisfactory (as it always happen on first iterations), you can set the expected output in its specific buffer and ask the network to backpropagate the error.

```obj-c
// Set the expected output
net.expectedOutputBuffer[0]= 0.5;

// Backpropagate the error
[net backPropagateWithLearningRate:0.1];
```

The network automatically computes the error and applies the gradient descent algorithm to obtain new weights. The *learning rate* parameter makes learning faster (and more uncertain) for greater values, or slower (but more certain) for lower values.

**New weights are not applied immediately**: they are stored inside the network, so that you may run multiple feed forwards and backpropagations before applying them (i.e. train by epochs).

Once your training is done, update the weights in the following way:

```obj-c
// Update the weights
[net updateWeights];
```

### Training loop example

You may update the weights after each backpropagation, and this is called *training by sample*, or wait until an entire training set has been performed, and this is called *training by epochs* (a training set is an "epoch", don't ask me why).

A typical training-by-sample loop is the following:

```obj-c
for (int i= 0; i < numberOfSets; i++) {

	// Load the i-th set of samples
	// ...

	for (int i= 0; i < numberOfSamples; i++) {
		ML_VDSP_VCLR(net.inputBuffer, 1, net.inputSize);

		// Load the j-th training sample
		net.inputBuffer[0]= 1.0;
		net.inputBuffer[2]= 0.5;
		// ...

		[net feedForward];

		// Check if output is correct
		if (net.outputBuffer[0] == 0.5) {
			matches++;

		} else {

			// Set the expected output
			net.expectedOutputBuffer[0]= 0.5;
			// ...

			[net backPropagateWithLearningRate:0.1];

			// Update weights for just this sample
			[net updateWeights];
		}
	}
}
```

While a typical training-by-epoch loop is the following:

```obj-c
for (int i= 0; i < numberOfSets; i++) {

	// Load the i-th set of samples
	// ...

	for (int j= 0; j < numberOfSamples; j++) {
		ML_VDSP_VCLR(net.inputBuffer, 1, net.inputSize);

		// Load the j-th training sample
		net.inputBuffer[0]= 1.0;
		net.inputBuffer[2]= 0.5;
		// ...

		[net feedForward];

		// Check if output is correct
		if (net.outputBuffer[0] == 0.5) {
			matches++;

		} else {

			// Set the expected output
			net.expectedOutputBuffer[0]= 0.5;
			// ...

			[net backPropagateWithLearningRate:0.1];
		}
	}

	// Update weights for all samples
	[net updateWeights];
}
```

At the end of the loop you usually check the number of matches and decide if the confidence level is high enough or not. If it is not, the training restarts from the beginning.

The network tries to enforce the correct calling sequence by using a simple state machine. Check the following state diagram:

![Network States](Network%20States.png)

If you try a call that does not correspond to state transition in the above diagram, the network will throw an exception.


## Choosing the appropriate model

This is the hardest part. Your neural network depends on a number of high-level parameters:

* dimension of the input layer (i.e. how you format the input data);
* number and dimensions of the hidden layers;
* type of activation function;
* training model (by sample or by epoch);
* learning rate.

Choosing the right values is quite difficult and is exactly why frameworks like this exist: so you can try locally before spending lots of money running the same network on a datacenter.

Some simple advices:

* the network may easily diverge if input data contains higher values (logistic and hyperbolic functions makes use of exponentials), **normalize your input data** in some way;
* **keep your model simple**: don't add hidden layers if they are not needed, they will slow down the network consistently and may not improve its prediction abilities;
* if your output is in range \[0..1\] (i.e. for **classification**), use the step or the logistic function, if you need an output in the range \[-1..1\] then move to the hyperbolic function, if you need a more wide-ranged output (i.e. for **regression**) use the linear function;
* start with **training by sample**, switch to training by epoch only if you know what you are doing;
* finally, **keep the learning rate small**: 0.1, or even lower.


## Examples

The library contains 3 unit tests that show how to use it, see [NeuralNetTests.m](MAChineLearningTests/NeuralNetTests.m) or [NeuralNetSwiftTests.swift](MAChineLearningTests/NeuralNetSwiftTests.m).

The first of them is the NAND logic port discussed on [Wikipedia](http://en.wikipedia.org/wiki/Perceptron#Example). The tests includes a few commented lines that, if uncommented, dump the network status after each training. Note that if you compare it with the Wikipedia example number will differ. This is due to the use of bias, which the Wikipedia example does not include.


## References

There are a lot articles out there explaining how neural networks work, but I have found these two in particular well written and clear enough to base my coding on them:

* [Machine Learning: Multi Layer Perceptrons](http://ml.informatik.uni-freiburg.de/_media/teaching/ss10/05_mlps.printer.pdf) [PDF]
* [Designing And Implementing A Neural Network Library For Handwriting Detection, Image Analysis etc.](http://www.codeproject.com/Articles/14342/Designing-And-Implementing-A-Neural-Network-Librar)

I am grateful to these people for taking the time to share their knowledge.




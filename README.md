# MAChineLearning
Machine Learning for the Mac


## Intro

This framework provides a quick and easy way to experiment with machine learning with native code on the Mac. It is written in Objective-C and should be usable with Swift as well (haven't tested).

The framework supports:

- single layer perceptrons;
- multilayer perceptrons of any depth (limited by memory);
- 4 kinds of activation function:
  - linear,
  - step (0 if output is less than 0.5, 1 if greater),
  - logistic,
  - hyperbolic;
- training through backpropagation with standard gradient descent, both by sample and by epoch;
- load/save of the network status;
- single/double precision (needs recompilation, default is single precision).

Internal code makes heavy use of vDSP and vecLib functions. It is as fast as it can get on a CPU, but not as fast as it could get with OpenCL on a GPU. An OpenCL version may come in the future.


## Use on iOS

The library should work well on iOS too. Ideally, one could write a program on the Mac to train the network, save its status and then load it on an iPhone or iPad and use it pre-trained. Haven't tested this scenario, but I see no reason it should not work.


## Examples

The library contains 3 unit/integration tests that show how to use it, including the [NAND](http://en.wikipedia.org/wiki/Perceptron#Example) use case discussed on Wikipedia. Anyway here is a quick tutorial.

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

Buffers are exposed as C arrays of type `nnREAL`, which by default is a typedef of `float`. You can redefine this type to a `double` in the `NeuralNetworkReal.h` file and then recompiling, just follow the comments.

**Be careful when reading and writing these arrays** since there's no bounds checking, you may easily cause an `EXC_BAD_ACCESS`.

### Loading input

So, this is how you load your input data:

```obj-c
// Clear the input buffer using vDSP
nnVDSP_VCLR(net.inputBuffer, 1, net.inputSize);

// Fill appropriate buffer elements
net.inputBuffer[0]= 1.0;
net.inputBuffer[1]= 0.0;
net.inputBuffer[2]= 0.0;
```

To clear the buffer with vDSP you need the import of Accelerate, shown above. Here we use the nnVDSP_VCLR macro, which is appropriately defined for the `nnREAL` type. Using Accelerate to clear the buffer is advised as it is quicker than a simple loop.

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
net.expectedOutputBuffer[0]= 1.0;

// Backpropagate the error
[net backPropagateWithLearningRate:0.1];
```

The network automatically computes the error and applies the gradient descent algorithm to obtain new weights. The *learning rate* parameter makes learning faster (and more uncertain) or slower (but more certain). More on this later.

**New weights are not applied immediately**: they are stored inside the network, so that you may run multiple feed forwards and backpropagations before applying them (i.e. train by epochs, see below).

Once your training is done, update the weights in the following way:

```obj-c
// Update the weights
[net updateWeights];
```

You may update the weights after each backpropagation, and this is called *training by sample*, or wait until an entire training set has been performed, and this is called *training by epochs* (a training set is an "epoch", don't ask me why).

To be more clear, a typical training by sample loop is the following:

```obj-c
for (int i= 0; i < numberOfSets; i++) {

    // Load the set of samples
    // ...

    for (int i= 0; i < numberOfSamples; i++) {

        // Load the current sample
        net.inputBuffer[0]= 1.0;
        // ...

        [net feedForward];

        // Set the expected output
        net.expectedOutputBuffer[0]= 1.0;
        // ...

        [net backPropagateWithLearningRate:0.1];
        [net updateWeights];
    }
}
```

While a typical training by epoch loop is the following:

```obj-c
for (int i= 0; i < numberOfSets; i++) {

    // Load the set of samples
    // ...

    for (int j= 0; j < numberOfSamples; j++) {

        // Load the current sample
        net.inputBuffer[0]= 1.0;
        // ...

        [net feedForward];

        // Set the expected output
        net.expectedOutputBuffer[0]= 1.0;
        // ...

        [net backPropagateWithLearningRate:0.1];
    }

    [net updateWeights];
}
```

The network object tries to enforce the correct calling sequence by keeping trace of a status. Check the following state diagram:

![Network States](Network%20States.png)

If you try a call that does not correspond to an arrow in the above diagram, the network object will throw an exception.


## Choosing the appropriate model

This is the hardest part. Your neural network depends on a number of  parameters:

* dimension of the input layer (i.e. how you format the input data);
* number and dimensions of the hidden layers;
* type of activation function;
* training model (by sample or by epoch);
* learning rate.

Choosing the right values is quite difficult and is exactly why frameworks like this exist: so you can try locally before spending lots of money running the same network on a datacenter.

Some simple advices from a beginner like me:

* generally speaking, let the network discover the pattern: bigger input layers slow down the network but may improve the results;
* start with simple models: don't add hidden layers if they are not needed, they will slow down the network consistently and may not improve its prediction abilities;
* if your output is in range \[0..1\] (i.e. for classification), the logistic function is your friend, if you need an output in the range \[-1..1\] then move to the hyperbolic function, but beware: it easily diverges when input values are greater than 1, apply some sort of normalization;
* start with training by sample, switch to training by epoch only if you know what you are doing;
* finally, keep the learning rate small: 0.1, or even lower; you may even dynamically lower it as the network approaches the expected results (i.e. higher learning rates with higher discrepancies, lower learning rates for lower discrepancies).


## Summing things up

I am a professional developer but not a data scientist. I wrote this library because, you know, they say you haven't really understood something until you code it. So, here it is. Use it to experiment and have fun, and if you find it useful I will be happy to hear it at [@foolish_dev](http://www.twitter.com/foolish_dev).

Remember, anyway, that it may contain bugs and/or conceptual mistakes. If you find any of them, please report. So that I can fix them and, most importantly, learn something new.

Enjoy.

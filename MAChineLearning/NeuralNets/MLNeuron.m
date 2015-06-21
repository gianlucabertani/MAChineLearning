//
//  MLNeuron.m
//  MAChineLearning
//
//  Created by Gianluca Bertani on 01/03/15.
//  Copyright (c) 2015 Gianluca Bertani. All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//  * Neither the name of Gianluca Bertani nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//

#import "MLNeuron.h"
#import "MLNeuronLayer.h"
#import "MLNeuralNetworkException.h"

#import "MLConstants.h"
#import "MLRandom.h"

#import <Accelerate/Accelerate.h>

#define DUMP_VECTOR(x) \
	{ \
		NSMutableString *dump= [[NSMutableString alloc] init]; \
		for (int i= 0; i < _inputSize; i++) \
			[dump appendFormat:@" %+8.2f |", x[i]]; \
		NSLog(@"%20s: %@", #x, dump); \
	}


#pragma mark -
#pragma mark Neuron extension

@interface MLNeuron () {
	MLNeuronLayer __weak *_layer;
	
	NSUInteger _index;
	MLReal *_outputBuffer;
	
	NSUInteger _inputSize;
	MLReal *_inputBuffer;

	MLReal *_weights;
	MLReal *_weightsDelta;
	
	MLReal **_nextLayerWeightPtrs;
	MLReal **_nextLayerWeightDeltaPtrs;

	// RPROP related
	MLReal *_weightSteps;
	MLReal *_previousGradient;
	MLReal *_previousWeightsChange;
	
	MLReal *_gradient;
	MLReal *_gradientSign;
	MLReal *_gradientsProduct;

	MLReal *_weightsRestore;
	MLReal *_rpropTemp;
}


#pragma mark -
#pragma mark Backpropagation internals

- (void) backPropagateWithLearningRate:(MLReal)learningRate delta:(MLReal)delta;
- (void) backPropagateResilientlyWithDelta:(MLReal)delta;


@end


#pragma mark -
#pragma mark Static constants

static const MLReal __minusEpsilon=          -1e-36;
static const MLReal __epsilon=                1e-36;
static const MLReal __one=                    1.0;
static const MLReal __minusHalf=             -0.5;
static const MLReal __half=                   0.5;
static const MLReal __zero=                   0.0;

static const MLReal __stepInitialValue=       0.1;
static const MLReal __stepAcceleration=       0.2;
static const MLReal __stepDeceleration=      -0.5;
static const MLReal __stepMin=                0.000001;
static const MLReal __stepMax=                1.0;
static const MLReal __stepScaleFactor=        1e+35;
static const MLReal __stepScaledAcceleration= __stepAcceleration / __stepScaleFactor;
static const MLReal __stepScaledDeceleration= __stepDeceleration / __stepScaleFactor;


#pragma mark -
#pragma mark Neuron implementation

@implementation MLNeuron


#pragma mark -
#pragma mark Initialization

- (instancetype) initWithLayer:(MLNeuronLayer *)layer index:(NSUInteger)index outputBuffer:(MLReal *)outputBuffer inputSize:(NSUInteger)inputSize inputBuffer:(MLReal *)inputBuffer {
	if ((self = [super init])) {
		
		// Initialization
		_layer= layer;
		
		_index= index;
		_outputBuffer= outputBuffer;
		
		_inputSize= inputSize;
		_inputBuffer= inputBuffer;
		
		// Allocate buffers
		int err= posix_memalign((void **) &_weights,
								BUFFER_MEMORY_ALIGNMENT,
								sizeof(MLReal) * _inputSize);
		if (err)
			@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating buffer"
																   userInfo:@{@"buffer": @"weights",
																			  @"error": [NSNumber numberWithInt:err]}];

		err= posix_memalign((void **) &_weightsDelta,
							BUFFER_MEMORY_ALIGNMENT,
							sizeof(MLReal) * _inputSize);
		if (err)
			@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating buffer"
																   userInfo:@{@"buffer": @"weightsDelta",
																			  @"error": [NSNumber numberWithInt:err]}];
		
		// Clear and fill buffers as needed
		ML_VDSP_VCLR(_weights, 1, _inputSize);
		ML_VDSP_VCLR(_weightsDelta, 1, _inputSize);
	}
	
	return self;
}

- (void) dealloc {

	// Deallocate weights and gradients
	free(_weights);
	_weights= NULL;
	
	free(_weightsDelta);
	_weightsDelta= NULL;

	// Deallocate pointes for weight gathering
	if (_nextLayerWeightPtrs) {
		free(_nextLayerWeightPtrs);
		_nextLayerWeightPtrs= NULL;
	}
	
	if (_nextLayerWeightDeltaPtrs) {
		free(_nextLayerWeightDeltaPtrs);
		_nextLayerWeightDeltaPtrs= NULL;
	}

	// Deallocate pointers for RPROP
	if (_weightSteps) {
		free(_weightSteps);
		_weightSteps= NULL;
	}
	
	if (_previousGradient) {
		free(_previousGradient);
		_previousGradient= NULL;
	}
	
	if (_previousWeightsChange) {
		free(_previousWeightsChange);
		_previousWeightsChange= NULL;
	}
	
	if (_gradient) {
		free(_gradient);
		_gradient= NULL;
	}
	
	if (_gradientSign) {
		free(_gradientSign);
		_gradientSign= NULL;
	}
	
	if (_gradientsProduct) {
		free(_gradientsProduct);
		_gradientsProduct= NULL;
	}
	
	if (_weightsRestore) {
		free(_weightsRestore);
		_weightsRestore= NULL;
	}

	if (_rpropTemp) {
		free(_rpropTemp);
		_rpropTemp= NULL;
	}
}


#pragma mark -
#pragma mark Setup and randomization

- (void) setUpForBackpropagationWithAlgorithm:(MLBackPropagationType)backPropType {
	if (_nextLayerWeightPtrs || _gradient)
		@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Neuron already set up"
																 userInfo:@{@"layer": [NSNumber numberWithUnsignedInteger:self.layer.index],
																			@"neuron": [NSNumber numberWithUnsignedInteger:self.index]}];
	
	if (self.layer.nextLayer) {
		MLNeuronLayer *nextLayer= (MLNeuronLayer *) self.layer.nextLayer;
		
		// Set up pointers to gather weights of next layer
		int err= posix_memalign((void **) &_nextLayerWeightPtrs,
								BUFFER_MEMORY_ALIGNMENT,
								sizeof(MLReal *) * nextLayer.size);
		if (err)
			@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating buffer"
																	 userInfo:@{@"buffer": @"nextLayerWeightPtrs",
																				@"error": [NSNumber numberWithInt:err]}];
		
		err= posix_memalign((void **) &_nextLayerWeightDeltaPtrs,
							BUFFER_MEMORY_ALIGNMENT,
							sizeof(MLReal *) * nextLayer.size);
		if (err)
			@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating buffer"
																	 userInfo:@{@"buffer": @"nextLayerWeightDeltaPtrs",
																				@"error": [NSNumber numberWithInt:err]}];
		
		// Fill pointers
		int j= 0;
		for (MLNeuron *nextNeuron in nextLayer.neurons) {
			_nextLayerWeightPtrs[j]= &(nextNeuron.weights[_index]);
			_nextLayerWeightDeltaPtrs[j]= &(nextNeuron.weightsDelta[_index]);
			j++;
		}
	}
	
	switch (backPropType) {
		case MLBackPropagationTypeRPROP: {
			
			// Allocate more buffers for RPROP
			int err= posix_memalign((void **) &_weightSteps,
									BUFFER_MEMORY_ALIGNMENT,
									sizeof(MLReal) * _inputSize);
			if (err)
				@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating buffer"
																		 userInfo:@{@"buffer": @"weightSteps",
																					@"error": [NSNumber numberWithInt:err]}];
			
			err= posix_memalign((void **) &_previousGradient,
								BUFFER_MEMORY_ALIGNMENT,
								sizeof(MLReal) * _inputSize);
			if (err)
				@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating buffer"
																		 userInfo:@{@"buffer": @"previousGradient",
																					@"error": [NSNumber numberWithInt:err]}];
			
			err= posix_memalign((void **) &_previousWeightsChange,
								BUFFER_MEMORY_ALIGNMENT,
								sizeof(MLReal) * _inputSize);
			if (err)
				@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating buffer"
																		 userInfo:@{@"buffer": @"previousWeightsChange",
																					@"error": [NSNumber numberWithInt:err]}];
			err= posix_memalign((void **) &_gradient,
								BUFFER_MEMORY_ALIGNMENT,
								sizeof(MLReal) * _inputSize);
			if (err)
				@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating buffer"
																		 userInfo:@{@"buffer": @"gradient",
																					@"error": [NSNumber numberWithInt:err]}];
			
			err= posix_memalign((void **) &_gradientSign,
								BUFFER_MEMORY_ALIGNMENT,
								sizeof(MLReal) * _inputSize);
			if (err)
				@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating buffer"
																		 userInfo:@{@"buffer": @"gradientSign",
																					@"error": [NSNumber numberWithInt:err]}];
			
			err= posix_memalign((void **) &_gradientsProduct,
								BUFFER_MEMORY_ALIGNMENT,
								sizeof(MLReal) * _inputSize);
			if (err)
				@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating buffer"
																		 userInfo:@{@"buffer": @"gradientsProduct",
																					@"error": [NSNumber numberWithInt:err]}];
			
			err= posix_memalign((void **) &_weightsRestore,
								BUFFER_MEMORY_ALIGNMENT,
								sizeof(MLReal) * _inputSize);
			if (err)
				@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating buffer"
																		 userInfo:@{@"buffer": @"weightsRestore",
																					@"error": [NSNumber numberWithInt:err]}];
			
			err= posix_memalign((void **) &_rpropTemp,
								BUFFER_MEMORY_ALIGNMENT,
								sizeof(MLReal) * _inputSize);
			if (err)
				@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating buffer"
																		 userInfo:@{@"buffer": @"rpropTemp",
																					@"error": [NSNumber numberWithInt:err]}];
			
			// Clear and fill buffers as needed
			ML_VDSP_VFILL(&__stepInitialValue, _weightSteps, 1, _inputSize);
			ML_VDSP_VCLR(_previousGradient, 1, _inputSize);
			ML_VDSP_VCLR(_previousWeightsChange, 1, _inputSize);
			
			break;
		}
			
		default:
			break;
	}
}

- (void) randomizeWeights {
	[MLRandom fillVector:_weights size:_inputSize ofGaussianRealsWithMean:0.0 sigma:ML_SQRT(_inputSize)];
}


#pragma mark -
#pragma mark Operations

- (void) feedForward {
	
	// Compute the dot product, the rest of the computation is done in the layer
	ML_VDSP_DOTPR(_inputBuffer, 1, _weights, 1, &_outputBuffer[_index], _inputSize);
}

- (void) backPropagateWithAlgorithm:(MLBackPropagationType)backPropType learningRate:(MLReal)learningRate delta:(MLReal)delta {
	switch (backPropType) {
		case MLBackPropagationTypeStandard:
			[self backPropagateWithLearningRate:learningRate delta:delta];
			break;
			
		case MLBackPropagationTypeRPROP:
			[self backPropagateResilientlyWithDelta:delta];
			break;
	}
}

- (void) updateWeights {
	
	// Add the weights with the weights delta
	ML_VDSP_VADD(_weightsDelta, 1, _weights, 1, _weights, 1, _inputSize);
	
	// Clear the weights delta buffer
	ML_VDSP_VCLR(_weightsDelta, 1, _inputSize);
}


#pragma mark -
#pragma mark Backpropagation internals

- (void) backPropagateWithLearningRate:(MLReal)learningRate delta:(MLReal)delta {
	
	// We receive the delta from the caller (instead of using self.delta) to avoid
	// a method call, which wastes lots of time
	MLReal deltaRate= learningRate * delta;
	
	// Compute weights delta using vector multiply & add,
	// the rest of the back propagation is done in the layer
	ML_VDSP_VSMA(_inputBuffer, 1, &deltaRate, _weightsDelta, 1, _weightsDelta, 1, _inputSize);
}

- (void) backPropagateResilientlyWithDelta:(MLReal)delta {
	
	// Compute the current gradient
	ML_VDSP_VSMUL(_inputBuffer, 1, &delta, _gradient, 1, _inputSize);
	
	// Compute the gradient sign: we have to apply an inverted clip to
	// ensure no division by zero will be performed
	// NOTE: VDIV operands are inverted compared to documentation (see function
	// definition for operand order)
	ML_VDSP_VICLIP(_gradient, 1, &__minusEpsilon, &__epsilon, _rpropTemp, 1, _inputSize);
	ML_VDSP_VABS(_rpropTemp, 1, _rpropTemp, 1, _inputSize);
	ML_VDSP_VDIV(_rpropTemp, 1, _gradient, 1, _gradientSign, 1, _inputSize);
	
	// Compute product of current gradient by previous gradient
	ML_VDSP_VMUL(_gradient, 1, _previousGradient, 1, _gradientsProduct, 1, _inputSize);
	
	// Use clipping functions to compute change factor of weight steps:
	// we use scaled down acceleration/deceleration factors as thresholds,
	// then scale up the resulting vector appropriately
	ML_VDSP_VCLIP(_gradientsProduct, 1, &__stepScaledDeceleration, &__stepScaledAcceleration, _rpropTemp, 1, _inputSize);
	ML_VDSP_VSMUL(_rpropTemp, 1, &__stepScaleFactor, _rpropTemp, 1, _inputSize);
	
	// Multiply and add the change factor to obtain final weights steps,
	// also apply a clip to ensure steps don't get too big or small
	ML_VDSP_VMA(_rpropTemp, 1, _weightSteps, 1, _weightSteps, 1, _weightSteps, 1, _inputSize);
	ML_VDSP_VCLIP(_weightSteps, 1, &__stepMin, &__stepMax, _weightSteps, 1, _inputSize);
	
	// Apply threshold and sum to gradients product to find which weights
	// must be reset to their previous value: this vector has -1 where the
	// gradients product is negative, 0 otherwise
	ML_VDSP_VTHRSC(_gradientsProduct, 1, &__zero, &__half, _rpropTemp, 1, _inputSize);
	ML_VDSP_VSADD(_rpropTemp, 1, &__minusHalf, _rpropTemp, 1, _inputSize);
	
	// Multiply for previous weight change, the result vector is the "restore" vector:
	// has 0 where the gradients product is positive, and the opposite of previous weight
	// change where the gradients product is negative
	ML_VDSP_VMUL(_rpropTemp, 1, _previousWeightsChange, 1, _weightsRestore, 1, _inputSize);
	
	// Complement the precursor of the "restore" vector: the result has 1
	// where the gradients product is positive, 0 otherwise
	ML_VDSP_VSADD(_rpropTemp, 1, &__one, _rpropTemp, 1, _inputSize);
	
	// Nullify the gradient where the gradients product is negative
	ML_VDSP_VMUL(_gradient, 1, _rpropTemp, 1, _gradient, 1, _inputSize);
	
	// Compute the final weights change by multiplying the weight steps by the complement
	// of the "restore" vector, then multuplying by the sign of the gradient and finally
	// adding with the "restore" vector
	ML_VDSP_VMUL(_weightSteps, 1, _rpropTemp, 1, _rpropTemp, 1, _inputSize);
	ML_VDSP_VMA(_rpropTemp, 1, _gradientSign, 1, _weightsRestore, 1, _rpropTemp, 1, _inputSize);
	
	// Finally apply the steps to weights delta
	ML_VDSP_VADD(_rpropTemp, 1, _weightsDelta, 1, _weightsDelta, 1, _inputSize);
		
	// Save gradient and weights delta for next step
	ML_VDSP_VSMUL(_gradient, 1, &__one, _previousGradient, 1, _inputSize);
	ML_VDSP_VSMUL(_rpropTemp, 1, &__one, _previousWeightsChange, 1, _inputSize);
}


#pragma mark -
#pragma mark Properties

@synthesize layer= _layer;

@synthesize index= _index;
@synthesize outputBuffer= _outputBuffer;

@synthesize inputSize= _inputSize;
@synthesize inputBuffer= _inputBuffer;

@dynamic bias;

- (MLReal) bias {
	return _layer.biasBuffer[_index];
}

- (void) setBias:(MLReal)bias {
	_layer.biasBuffer[_index]= bias;
}

@synthesize weights= _weights;
@synthesize weightsDelta= _weightsDelta;

@synthesize nextLayerWeightPtrs= _nextLayerWeightPtrs;
@synthesize nextLayerWeightDeltaPtrs= _nextLayerWeightDeltaPtrs;

@dynamic error;

- (MLReal) error {
	return _layer.errorBuffer[_index];
}

- (void) setError:(MLReal)error {
	_layer.errorBuffer[_index]= error;
}

@dynamic delta;

- (MLReal) delta {
	return _layer.deltaBuffer[_index];
}

- (void) setDelta:(MLReal)delta {
	_layer.deltaBuffer[_index]= delta;
}


@end

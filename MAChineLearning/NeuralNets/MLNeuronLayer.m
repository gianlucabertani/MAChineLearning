//
//  MLNeuronLayer.m
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

#import "MLNeuronLayer.h"
#import "MLInputLayer.h"
#import "MLNeuron.h"
#import "MLNeuralNetworkException.h"

#import "MLConstants.h"

#import <Accelerate/Accelerate.h>


#pragma mark -
#pragma mark NeuronLayer extension

@interface MLNeuronLayer () {
	MLActivationFunctionType _funcType;

	MLReal *_outputBuffer;
	
	MLReal *_biasBuffer;
	MLReal *_biasDeltaBuffer;

	MLReal *_deltaBuffer;
	MLReal *_errorBuffer;
	
	MLReal *_tempBuffer;
	MLReal *_nextLayerWeightsBuffer;

	NSMutableArray *_neurons;
	
	MLReal *_minusTwo;
	MLReal *_minusOne;
	MLReal *_zero;
	MLReal *_half;
	MLReal *_one;
}


@end


#pragma mark -
#pragma mark NeuronLayer implementation

@implementation MLNeuronLayer


#pragma mark -
#pragma mark Initialization

- (id) initWithIndex:(int)index size:(int)size activationFunctionType:(MLActivationFunctionType)funcType {
	if ((self = [super initWithIndex:index size:size])) {
		
		// Initialization
		_funcType= funcType;

		// Allocate buffers
		int err= posix_memalign((void **) &_outputBuffer,
								BUFFER_MEMORY_ALIGNMENT,
								sizeof(MLReal) * size);
		if (err)
			@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating buffer"
																   userInfo:@{@"buffer": @"outputBuffer",
																			  @"error": [NSNumber numberWithInt:err]}];
		
		err= posix_memalign((void **) &_biasBuffer,
							BUFFER_MEMORY_ALIGNMENT,
							sizeof(MLReal) * size);
		if (err)
			@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating buffer"
																   userInfo:@{@"buffer": @"biasBuffer",
																			  @"error": [NSNumber numberWithInt:err]}];

		err= posix_memalign((void **) &_biasDeltaBuffer,
							BUFFER_MEMORY_ALIGNMENT,
							sizeof(MLReal) * size);
		if (err)
			@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating buffer"
																   userInfo:@{@"buffer": @"biasDeltaBuffer",
																			  @"error": [NSNumber numberWithInt:err]}];

		err= posix_memalign((void **) &_deltaBuffer,
							BUFFER_MEMORY_ALIGNMENT,
							sizeof(MLReal) * size);
		if (err)
			@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating buffer"
																   userInfo:@{@"buffer": @"deltaBuffer",
																			  @"error": [NSNumber numberWithInt:err]}];
		
		err= posix_memalign((void **) &_errorBuffer,
							BUFFER_MEMORY_ALIGNMENT,
							sizeof(MLReal) * size);
		if (err)
			@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating buffer"
																   userInfo:@{@"buffer": @"errorBuffer",
																			  @"error": [NSNumber numberWithInt:err]}];
		
		err= posix_memalign((void **) &_tempBuffer,
							BUFFER_MEMORY_ALIGNMENT,
							sizeof(MLReal) * size);
		if (err)
			@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating buffer"
																   userInfo:@{@"buffer": @"tempBuffer",
																			  @"error": [NSNumber numberWithInt:err]}];
		
		// Clear and fill buffers as needed
		ML_VDSP_VCLR(_outputBuffer, 1, size);
		ML_VDSP_VCLR(_biasBuffer, 1, size);
		ML_VDSP_VCLR(_biasDeltaBuffer, 1, size);
		ML_VDSP_VCLR(_deltaBuffer, 1, size);
		ML_VDSP_VCLR(_errorBuffer, 1, size);
		ML_VDSP_VCLR(_tempBuffer, 1, size);
		
		// Allocate constants
		err= posix_memalign((void **) &_minusTwo,
							BUFFER_MEMORY_ALIGNMENT,
							sizeof(MLReal));
		if (err)
			@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating constant"
																   userInfo:@{@"constant": @"minusTwo",
																			  @"error": [NSNumber numberWithInt:err]}];

		err= posix_memalign((void **) &_minusOne,
							BUFFER_MEMORY_ALIGNMENT,
							sizeof(MLReal));
		if (err)
			@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating constant"
																   userInfo:@{@"constant": @"minusOne",
																			  @"error": [NSNumber numberWithInt:err]}];
		
		err= posix_memalign((void **) &_zero,
							BUFFER_MEMORY_ALIGNMENT,
							sizeof(MLReal));
		if (err)
			@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating constant"
																   userInfo:@{@"constant": @"zero",
																			  @"error": [NSNumber numberWithInt:err]}];

		err= posix_memalign((void **) &_half,
							BUFFER_MEMORY_ALIGNMENT,
							sizeof(MLReal));
		if (err)
			@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating constant"
																   userInfo:@{@"constant": @"half",
																			  @"error": [NSNumber numberWithInt:err]}];

		err= posix_memalign((void **) &_one,
							BUFFER_MEMORY_ALIGNMENT,
							sizeof(MLReal));
		if (err)
			@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating constant"
																   userInfo:@{@"constant": @"one",
																			  @"error": [NSNumber numberWithInt:err]}];

		
		// Initialize constants
		*_minusTwo= -2.0;
		*_minusOne= -1.0;
		*_zero= 0.0;
		*_half= 0.5;
		*_one= 1.0;
	}
	
	return self;
}

- (void) dealloc {
	
	// Deallocate buffers
	free(_outputBuffer);
	_outputBuffer= NULL;
	
	free(_biasBuffer);
	_biasBuffer= NULL;

	free(_biasDeltaBuffer);
	_biasDeltaBuffer= NULL;

	free(_deltaBuffer);
	_deltaBuffer= NULL;

	free(_errorBuffer);
	_errorBuffer= NULL;

	free(_tempBuffer);
	_tempBuffer= NULL;
	
	if (_nextLayerWeightsBuffer) {
		free(_nextLayerWeightsBuffer);
		_nextLayerWeightsBuffer= NULL;
	}
	
	free(_minusTwo);
	_minusTwo= NULL;
	
	free(_minusOne);
	_minusOne= NULL;
	
	free(_zero);
	_zero= NULL;
	
	free(_half);
	_half= NULL;
	
	free(_one);
	_one= NULL;
}


#pragma mark -
#pragma mark Operations

- (void) setUp {
	if (_neurons)
		@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Neuron layer already set up"
															   userInfo:nil];
	
	_neurons= [[NSMutableArray alloc] initWithCapacity:self.size];
	
	for (int i= 0; i < self.size; i++) {
		MLReal *inputBuffer= NULL;

		if ([self.previousLayer isKindOfClass:[MLInputLayer class]]) {
			inputBuffer= [(MLInputLayer *) self.previousLayer inputBuffer];

		} else if ([self.previousLayer isKindOfClass:[MLNeuronLayer class]]) {
			inputBuffer= [(MLNeuronLayer *) self.previousLayer outputBuffer];
		
		} else
			@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Unknown type of layer found as previous layer"
																   userInfo:@{@"previousLayer": self.previousLayer}];

		MLNeuron *neuron= [[MLNeuron alloc] initWithLayer:self
												index:i
										 outputBuffer:self.outputBuffer
											inputSize:self.previousLayer.size
										  inputBuffer:inputBuffer];
		
		[_neurons addObject:neuron];
	}

	if (self.nextLayer) {
		
		// Prepare buffer for weights of next layer
		MLNeuronLayer *nextLayer= (MLNeuronLayer *) self.nextLayer;
		
		int err= posix_memalign((void **) &_nextLayerWeightsBuffer,
								BUFFER_MEMORY_ALIGNMENT,
								sizeof(MLReal) * nextLayer.size);
		if (err)
			@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating buffer"
																   userInfo:@{@"buffer": @"nextLayerWeightsBuffer",
																			  @"error": [NSNumber numberWithInt:err]}];
		
		ML_VDSP_VCLR(_nextLayerWeightsBuffer, 1, nextLayer.size);
	}
}

- (void) feedForward {
	if (!_neurons)
		@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Neuron layer not yet set up"
															   userInfo:nil];
	
	// Reset error and delta
	ML_VDSP_VCLR(_deltaBuffer, 1, _size);
	ML_VDSP_VCLR(_errorBuffer, 1, _size);

	// First step: compute dot product for each neuron,
	// will fill the output buffer
	for (MLNeuron *neuron in _neurons)
		[neuron partialFeedForward];
	
	// Second step: add bias
	ML_VDSP_VADD(_outputBuffer, 1, _biasBuffer, 1, _outputBuffer, 1, _size);
	
	// Third step: apply activation function
	switch (_funcType) {
		case MLActivationFunctionTypeLinear:
			
			// Apply formula: output[i] = output[i]
			break;
			
		case MLActivationFunctionTypeStep:

			// Apply formula: output[i] = (output[i] < 0.5 ? 0.0 : 1.0)
			ML_VDSP_VTHRSC(_outputBuffer, 1, _half, _one, _tempBuffer, 1, _size);
			ML_VDSP_VTHRES(_tempBuffer, 1, _zero, _outputBuffer, 1, _size);
			break;
			
		case MLActivationFunctionTypeLogistic:
			
			// Apply formula: output[i] = 1 / (1 + exp(-output[i])
			ML_VDSP_VSMUL(_outputBuffer, 1, _minusOne, _tempBuffer, 1, _size);
			ML_VVEXP(_tempBuffer, _tempBuffer, &_size);
			ML_VDSP_VSADD(_tempBuffer, 1, _one, _tempBuffer, 1, _size);
			ML_VDSP_SVDIV(_one, _tempBuffer, 1, _outputBuffer, 1, _size);
			break;
			
		case MLActivationFunctionTypeHyperbolic: {

			// Apply formula: output[i] = (1 - exp(-2 * output[i])) / (1 + exp(-2 * output[i]))
			// Equivalent to: output[i] = tanh(output[i])
			// NOTE: VDIV operands are inverted compared to documentation (see function
			// definition for operand order)
			ML_VDSP_VSMUL(_outputBuffer, 1, _minusTwo, _tempBuffer, 1, _size);
			ML_VVEXP(_tempBuffer, _tempBuffer, &_size);
			ML_VDSP_VSADD(_tempBuffer, 1, _one, _outputBuffer, 1, _size);
			ML_VDSP_VSMUL(_tempBuffer, 1, _minusOne, _tempBuffer, 1, _size);
			ML_VDSP_VSADD(_tempBuffer, 1, _one, _tempBuffer, 1, _size);
			ML_VDSP_VDIV(_outputBuffer, 1, _tempBuffer, 1, _outputBuffer, 1, _size);
			break;
		}
	}
}

- (void) fetchErrorFromNextLayer {
	if (!_neurons)
		@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Neuron layer not yet set up"
															   userInfo:nil];
	
	MLNeuronLayer *nextLayer= (MLNeuronLayer *) self.nextLayer;
	for (int i= 0; i < _size; i++) {
		
		// Prepare the vector of weights
		int j= 0;
		for (MLNeuron *nextNeuron in nextLayer.neurons) {
			_nextLayerWeightsBuffer[j]= nextNeuron.weights[i] + nextNeuron.weightsDelta[i];
			j++;
		}
		
		// Compute the dot product
		ML_VDSP_DOTPR(nextLayer.deltaBuffer, 1, _nextLayerWeightsBuffer, 1, &_errorBuffer[i], nextLayer.size);
	}
}

- (void) backPropagateWithLearningRate:(MLReal)learningRate {
	if (!_neurons)
		@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Neuron layer not yet set up"
															   userInfo:nil];
	
	// First step: compute the delta with
	// activation function derivative
	switch (_funcType) {
		case MLActivationFunctionTypeLinear:
			
			// Apply formula: delta[i] = error[i]
			ML_VDSP_VSMUL(_errorBuffer, 1, _one, _deltaBuffer, 1, _size);
			break;
			
		case MLActivationFunctionTypeStep:
			if (self.nextLayer)
				@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Can't backpropagate in a hidden layer with step function"
																	   userInfo:nil];

			// Apply formula: delta[i] = error[i]
			ML_VDSP_VSMUL(_errorBuffer, 1, _one, _deltaBuffer, 1, _size);
			break;
			
		case MLActivationFunctionTypeLogistic:
			
			// Apply formula: delta[i] = output[i] * (1 - output[i]) * error[i]
			ML_VDSP_VSMUL(_outputBuffer, 1, _minusOne, _tempBuffer, 1, _size);
			ML_VDSP_VSADD(_tempBuffer, 1, _one, _tempBuffer, 1, _size);
			ML_VDSP_VMUL(_tempBuffer, 1, _outputBuffer, 1, _tempBuffer, 1, _size);
			ML_VDSP_VMUL(_tempBuffer, 1, _errorBuffer, 1, _deltaBuffer, 1, _size);
			break;
			
		case MLActivationFunctionTypeHyperbolic:
			
			// Apply formula: delta[i] = (1 - (output[i] * output[i])) * error[i]
			ML_VDSP_VSQ(_outputBuffer, 1, _tempBuffer, 1, _size);
			ML_VDSP_VSMUL(_tempBuffer, 1, _minusOne, _tempBuffer, 1, _size);
			ML_VDSP_VSADD(_tempBuffer, 1, _one, _tempBuffer, 1, _size);
			ML_VDSP_VMUL(_tempBuffer, 1, _errorBuffer, 1, _deltaBuffer, 1, _size);
			break;
	}

	// Second step: compute the bias delta
	ML_VDSP_VSMA(_deltaBuffer, 1, &learningRate, _biasDeltaBuffer, 1, _biasDeltaBuffer, 1, _size);
	
	// Third step: compute new weights for each neuron
	int i= 0;
	for (MLNeuron *neuron in _neurons) {
		[neuron partialBackPropagateWithLearningRate:learningRate delta:_deltaBuffer[i]];
		i++;
	}
}

- (void) updateWeights {
	if (!_neurons)
		@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Neuron layer not yet set up"
															   userInfo:nil];
	
	// First step: update the bias with the bias delta
	ML_VDSP_VADD(_biasBuffer, 1, _biasDeltaBuffer, 1, _biasBuffer, 1, _size);
	
	// Second step: update weights for each neuron
	for (MLNeuron *neuron in _neurons)
		[neuron partialUpdateWeights];

	// Clear the bias delta buffer
	ML_VDSP_VCLR(_biasDeltaBuffer, 1, _size);
}


#pragma mark -
#pragma mark Properties

@synthesize funcType= _funcType;

@synthesize biasBuffer= _biasBuffer;
@synthesize biasDeltaBuffer= _biasDeltaBuffer;

@synthesize errorBuffer= _errorBuffer;
@synthesize deltaBuffer= _deltaBuffer;

@synthesize outputBuffer= _outputBuffer;

@synthesize neurons= _neurons;


@end

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

#import <Accelerate/Accelerate.h>


#pragma mark -
#pragma mark Neuron extension

@interface MLNeuron () {
	MLNeuronLayer __weak *_layer;
	
	int _index;
	MLReal *_outputBuffer;
	
	int _inputSize;
	MLReal *_inputBuffer;

	MLReal *_weights;
	MLReal *_weightsDelta;
}

@end


#pragma mark -
#pragma mark Neuron implementation

@implementation MLNeuron


#pragma mark -
#pragma mark Initialization

- (id) initWithLayer:(MLNeuronLayer *)layer index:(int)index outputBuffer:(MLReal *)outputBuffer inputSize:(int)inputSize inputBuffer:(MLReal *)inputBuffer {
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

	// Deallocate the weights
	free(_weights);
	_weights= NULL;
	
	free(_weightsDelta);
	_weightsDelta= NULL;
}


#pragma mark -
#pragma mark Operations

- (void) partialFeedForward {
	
	// Compute the dot product, the rest of the computation is done in the layer
	ML_VDSP_DOTPR(_inputBuffer, 1, _weights, 1, &_outputBuffer[_index], _inputSize);
}

- (void) partialBackPropagateWithLearningRate:(MLReal)learningRate delta:(MLReal)delta {
	
	// We receive the delta from the caller (instead of using self.delta) to avoid
	// a method call, which wastes lots of time
	MLReal deltaRate= learningRate * delta;

	// Compute weights delta using vector multiply & add,
	// the rest of the back propagation is done in the layer
	ML_VDSP_VSMA(_inputBuffer, 1, &deltaRate, _weightsDelta, 1, _weightsDelta, 1, _inputSize);
}

- (void) partialUpdateWeights {
	
	// Add the weights with the weights delta
	ML_VDSP_VADD(_weightsDelta, 1, _weights, 1, _weights, 1, _inputSize);
	
	// Clear the weights delta buffer
	ML_VDSP_VCLR(_weightsDelta, 1, _inputSize);
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

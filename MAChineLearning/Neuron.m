//
//  Neuron.m
//  MAChineLearning
//
//  Created by Gianluca Bertani on 01/03/15.
//  Copyright (c) 2015 Flying Dolphin Studio. All rights reserved.
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

#import "Neuron.h"
#import "NeuronLayer.h"
#import "NeuralNetworkException.h"

#import <Accelerate/Accelerate.h>

#define NEURAL_NET_MEMORY_ALIGNMENT          (128)


#pragma mark -
#pragma mark Neuron extension

@interface Neuron () {
	NeuronLayer __weak *_layer;
	
	int _index;
	nnREAL *_outputBuffer;
	
	int _inputSize;
	nnREAL *_inputBuffer;

	nnREAL *_weights;
	nnREAL *_weightsDelta;
}

@end


#pragma mark -
#pragma mark Neuron implementation

@implementation Neuron


#pragma mark -
#pragma mark Initialization

- (id) initWithLayer:(NeuronLayer *)layer index:(int)index outputBuffer:(nnREAL *)outputBuffer inputSize:(int)inputSize inputBuffer:(nnREAL *)inputBuffer {
	if ((self = [super init])) {
		
		// Initialization
		_layer= layer;
		
		_index= index;
		_outputBuffer= outputBuffer;
		
		_inputSize= inputSize;
		_inputBuffer= inputBuffer;
		
		// Allocate buffers
		int err= posix_memalign((void **) &_weights,
								NEURAL_NET_MEMORY_ALIGNMENT,
								sizeof(nnREAL) * _inputSize);
		if (err)
			@throw [NeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating buffer"
																   userInfo:@{@"buffer": @"weights",
																			  @"error": [NSNumber numberWithInt:err]}];

		err= posix_memalign((void **) &_weightsDelta,
							NEURAL_NET_MEMORY_ALIGNMENT,
							sizeof(nnREAL) * _inputSize);
		if (err)
			@throw [NeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating buffer"
																   userInfo:@{@"buffer": @"weightsDelta",
																			  @"error": [NSNumber numberWithInt:err]}];
		
		// Clear and fill buffers as needed
		nnVDSP_VCLR(_weights, 1, _inputSize);
		nnVDSP_VCLR(_weightsDelta, 1, _inputSize);
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
#pragma mark Operation

- (void) partialFeedForward {
	
	// Compute the dot product, the rest of the computation is done in the layer
	nnVDSP_DOTPR(_inputBuffer, 1, _weights, 1, &_outputBuffer[_index], _inputSize);
}

- (void) partialBackPropagateWithLearningRate:(nnREAL)learningRate delta:(nnREAL)delta {
	
	// We receive the delta from the caller (instead of using self.delta) to avoid
	// a method call, which wastes lots of time
	nnREAL deltaRate= learningRate * delta;

	// Compute weights delta using vector multiply & add,
	// the rest of the back propagation is done in the layer
	nnVDSP_VSMA(_inputBuffer, 1, &deltaRate, _weightsDelta, 1, _weightsDelta, 1, _inputSize);
}

- (void) partialUpdateWeights {
	
	// Add the weights with the weights delta
	nnVDSP_VADD(_weightsDelta, 1, _weights, 1, _weights, 1, _inputSize);
	
	// Clear the weights delta buffer
	nnVDSP_VCLR(_weightsDelta, 1, _inputSize);
}


#pragma mark -
#pragma mark Properties

@synthesize layer= _layer;

@synthesize index= _index;
@synthesize outputBuffer= _outputBuffer;

@synthesize inputSize= _inputSize;
@synthesize inputBuffer= _inputBuffer;

@dynamic bias;

- (nnREAL) bias {
	return _layer.biasBuffer[_index];
}

- (void) setBias:(nnREAL)bias {
	_layer.biasBuffer[_index]= bias;
}

@synthesize weights= _weights;
@synthesize weightsDelta= _weightsDelta;

@dynamic error;

- (nnREAL) error {
	return _layer.errorBuffer[_index];
}

- (void) setError:(nnREAL)error {
	_layer.errorBuffer[_index]= error;
}

@dynamic delta;

- (nnREAL) delta {
	return _layer.deltaBuffer[_index];
}

- (void) setDelta:(nnREAL)delta {
	_layer.deltaBuffer[_index]= delta;
}


@end

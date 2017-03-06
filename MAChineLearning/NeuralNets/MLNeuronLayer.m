//
//  MLNeuronLayer.m
//  MAChineLearning
//
//  Created by Gianluca Bertani on 01/03/15.
//  Copyright (c) 2015-2017 Gianluca Bertani. All rights reserved.
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
#import "MLBiasNeuron.h"
#import "MLNeuralNetworkException.h"

#import "MLAlloc.h"

#define DUMP_VECTOR(x) \
	{ \
		NSMutableString *dump= [[NSMutableString alloc] init]; \
		for (int i= 0; i < _size; i++) \
			[dump appendFormat:@" %+8.2f |", x[i]]; \
		NSLog(@"%20s: %@", #x, dump); \
	}


#pragma mark -
#pragma mark NeuronLayer extension

@interface MLNeuronLayer () {
	MLActivationFunctionType _funcType;

	MLReal *_outputBuffer;
	
	MLReal *_deltaBuffer;
	MLReal *_errorBuffer;
	
	MLReal *_nextLayerWeightsBuffer;
	MLReal *_nextLayerWeightsDeltaBuffer;

	BOOL _usingBias;
	NSMutableArray *_neurons;
}


@end


#pragma mark -
#pragma mark Static constants

static const MLReal __minusFourty= -40.0;
static const MLReal __minusTwo=     -2.0;
static const MLReal __minusOne=     -1.0;
static const MLReal __zero=          0.0;
static const MLReal __half=          0.5;
static const MLReal __one=           1.0;
static const MLReal __fourty=       40.0;


#pragma mark -
#pragma mark NeuronLayer implementation

@implementation MLNeuronLayer


#pragma mark -
#pragma mark Initialization

- (instancetype) initWithIndex:(NSUInteger)index size:(NSUInteger)size useBias:(BOOL)useBias activationFunctionType:(MLActivationFunctionType)funcType {
	if ((self = [super initWithIndex:index size:(useBias ? (size +1) : size)])) {
		
		// Initialization
		_funcType= funcType;
		_usingBias= useBias;
	}
	
	return self;
}

- (void) dealloc {
	
	// Deallocate buffers
	MLFreeRealBuffer(_outputBuffer);
	_outputBuffer= NULL;
	
	MLFreeRealBuffer(_deltaBuffer);
	_deltaBuffer= NULL;

	MLFreeRealBuffer(_errorBuffer);
	_errorBuffer= NULL;

    MLFreeRealBuffer(_nextLayerWeightsBuffer);
    _nextLayerWeightsBuffer= NULL;

    MLFreeRealBuffer(_nextLayerWeightsDeltaBuffer);
    _nextLayerWeightsDeltaBuffer= NULL;
}


#pragma mark -
#pragma mark Setup and randomization

- (void) setUp {
	if (_neurons)
		@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Neuron layer already set up"
																 userInfo:@{@"layer": [NSNumber numberWithUnsignedInteger:self.index]}];
	
	// Allocate buffers
    _outputBuffer= MLAllocRealBuffer(self.size);
    _deltaBuffer= MLAllocRealBuffer(self.size);
    _errorBuffer= MLAllocRealBuffer(self.size);
	
	// Clear and fill buffers as needed
	ML_VCLR(_outputBuffer, 1, self.size);
	ML_VCLR(_deltaBuffer, 1, self.size);
	ML_VCLR(_errorBuffer, 1, self.size);

	_neurons= [[NSMutableArray alloc] initWithCapacity:self.size];
	
	for (int i= 0; i < self.size; i++) {
		MLReal *inputBuffer= NULL;

		if ([self.previousLayer isKindOfClass:[MLInputLayer class]]) {
			inputBuffer= [(MLInputLayer *) self.previousLayer inputBuffer];

		} else if ([self.previousLayer isKindOfClass:[MLNeuronLayer class]]) {
			inputBuffer= [(MLNeuronLayer *) self.previousLayer outputBuffer];
		
		} else
			@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Unknown type of layer found as previous layer"
																	 userInfo:@{@"layer": [NSNumber numberWithUnsignedInteger:self.index],
																				@"previousLayer": [NSNumber numberWithUnsignedInteger:self.previousLayer.index]}];

		MLNeuron *neuron= nil;
		if (_usingBias && (i == (self.size -1))) {
			
			// Create a bias neurson
			neuron= [[MLBiasNeuron alloc] initWithLayer:self
												  index:i
										   outputBuffer:self.outputBuffer
											  inputSize:self.previousLayer.size
											inputBuffer:inputBuffer];
			
		} else {
			
			// Create standard neuron
			neuron= [[MLNeuron alloc] initWithLayer:self
											  index:i
									   outputBuffer:self.outputBuffer
										  inputSize:self.previousLayer.size
										inputBuffer:inputBuffer];
		}
		
		[_neurons addObject:neuron];
	}

	if (self.nextLayer) {
		
		// Prepare buffer for weights of next layer
		MLNeuronLayer *nextLayer= (MLNeuronLayer *) self.nextLayer;
		
        _nextLayerWeightsBuffer= MLAllocRealBuffer(nextLayer.size);
        _nextLayerWeightsDeltaBuffer= MLAllocRealBuffer(nextLayer.size);
		
		ML_VCLR(_nextLayerWeightsBuffer, 1, nextLayer.size);
		ML_VCLR(_nextLayerWeightsDeltaBuffer, 1, nextLayer.size);
	}
}

- (void) randomizeWeights {
	if (!_neurons)
		@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Neuron layer not yet set up"
																 userInfo:@{@"layer": [NSNumber numberWithUnsignedInteger:self.index]}];

	// Compute beta for Nguyen-Widrow randomization
	MLReal beta= 0.7 * ML_POW(((MLReal) self.size), 1.0 / ((MLReal) self.previousLayer.size));

	// Randomize each neuron
    for (MLNeuron *neuron in _neurons)
        [neuron randomizeWeightsWithBeta:beta];
}


#pragma mark -
#pragma mark Operations

- (void) feedForward {
	if (!_neurons)
		@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Neuron layer not yet set up"
																 userInfo:@{@"layer": [NSNumber numberWithUnsignedInteger:self.index]}];
	
	// Reset error and delta
	ML_VCLR(_deltaBuffer, 1, _size);
	ML_VCLR(_errorBuffer, 1, _size);

	// First step: compute dot product for each neuron,
	// will fill the output buffer
    for (MLNeuron *neuron in _neurons)
		[neuron feedForward];
	
	// Second step: apply activation function
	switch (_funcType) {
        case MLActivationFunctionTypeLinear: {
			
			// Apply formula: output[i] = output[i]
			break;
        }
			
        case MLActivationFunctionTypeRectifiedLinear: {
            
            // Apply formula: output[i] = (output[i] < 0.0 ? 0.0 : output[i])
            ML_VTHRES(_outputBuffer, 1, &__zero, _outputBuffer, 1, _size);
            break;
        }

        case MLActivationFunctionTypeStep: {
            MLReal *tempBuffer= MLAllocRealBuffer(self.size);

			// Apply formula: output[i] = (output[i] < 0.5 ? 0.0 : 1.0)
			ML_VTHRSC(_outputBuffer, 1, &__half, &__one, tempBuffer, 1, _size);
			ML_VTHRES(tempBuffer, 1, &__zero, _outputBuffer, 1, _size);
            
            MLFreeRealBuffer(tempBuffer);
			break;
        }
			
		case MLActivationFunctionTypeSigmoid: {
            MLReal *tempBuffer= MLAllocRealBuffer(self.size);
			
			// Apply clipping before the function to avoid NaNs
			ML_VCLIP(_outputBuffer, 1, &__minusFourty, &__fourty, _outputBuffer, 1, _size);
			
			// An "int" size is needed by vvexp,
			// the others still use _size
			int size= (int) _size;
			
			// Apply formula: output[i] = 1 / (1 + exp(-output[i])
			ML_VSMUL(_outputBuffer, 1, &__minusOne, tempBuffer, 1, _size);
			ML_VVEXP(tempBuffer, tempBuffer, &size);
			ML_VSADD(tempBuffer, 1, &__one, tempBuffer, 1, _size);
			ML_SVDIV(&__one, tempBuffer, 1, _outputBuffer, 1, _size);
            
            MLFreeRealBuffer(tempBuffer);
			break;
		}
			
		case MLActivationFunctionTypeTanH: {
            MLReal *tempBuffer= MLAllocRealBuffer(self.size);
			
			// Apply clipping before the function to avoid NaNs
			ML_VCLIP(_outputBuffer, 1, &__minusFourty, &__fourty, _outputBuffer, 1, _size);

			// An "int" size is needed by vvexp,
			// the others still use _size
			int size= (int) _size;

			// Apply formula: output[i] = (1 - exp(-2 * output[i])) / (1 + exp(-2 * output[i]))
			// Equivalent to: output[i] = tanh(output[i])
			ML_VSMUL(_outputBuffer, 1, &__minusTwo, tempBuffer, 1, _size);
			ML_VVEXP(tempBuffer, tempBuffer, &size);
			ML_VSADD(tempBuffer, 1, &__one, _outputBuffer, 1, _size);
			ML_VSMUL(tempBuffer, 1, &__minusOne, tempBuffer, 1, _size);
			ML_VSADD(tempBuffer, 1, &__one, tempBuffer, 1, _size);
			ML_VDIV(_outputBuffer, 1, tempBuffer, 1, _outputBuffer, 1, _size);
            
            MLFreeRealBuffer(tempBuffer);
			break;
		}
	}
}

- (void) fetchErrorFromNextLayer {
	if (!_neurons)
		@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Neuron layer not yet set up"
																 userInfo:@{@"layer": [NSNumber numberWithUnsignedInteger:self.index]}];
	
	MLNeuronLayer *nextLayer= (MLNeuronLayer *) self.nextLayer;
	
    for (MLNeuron *neuron in _neurons) {
		if ([neuron isKindOfClass:[MLBiasNeuron class]]) {
			
			// Bias neurons have constant output and don't backpropagate
			_errorBuffer[neuron.index]= __zero;
			
		} else {
			if ((!neuron.nextLayerWeightPtrs) || (!neuron.nextLayerWeightDeltaPtrs))
				@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Neuron not yet set up"
																		 userInfo:@{@"layer": [NSNumber numberWithUnsignedInteger:self.index],
																					@"neuron": [NSNumber numberWithUnsignedInteger:neuron.index]}];
			
			// Gather next layer weights using vector gathering
			ML_VGATHRA((const MLReal **) neuron.nextLayerWeightPtrs, 1, _nextLayerWeightsBuffer, 1, nextLayer.size);
			ML_VGATHRA((const MLReal **) neuron.nextLayerWeightDeltaPtrs, 1, _nextLayerWeightsDeltaBuffer, 1, nextLayer.size);
			
			// Sum the delta
			ML_VADD(_nextLayerWeightsBuffer, 1, _nextLayerWeightsDeltaBuffer, 1, _nextLayerWeightsBuffer, 1, nextLayer.size);
			
			// Compute the dot product
			ML_DOTPR(nextLayer.deltaBuffer, 1, _nextLayerWeightsBuffer, 1, &_errorBuffer[neuron.index], nextLayer.size);
		}
    }
}

- (void) backPropagateWithAlgorithm:(MLBackPropagationType)backPropType learningRate:(MLReal)learningRate costFunction:(MLCostFunctionType)costType {
	if (!_neurons)
		@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Neuron layer not yet set up"
																 userInfo:@{@"layer": [NSNumber numberWithUnsignedInteger:self.index]}];
	
	// First step: compute the delta with
	// activation function derivative
	switch (_funcType) {
        case MLActivationFunctionTypeLinear: {
			
			// Apply formula: delta[i] = error[i]
			ML_VSMUL(_errorBuffer, 1, &__one, _deltaBuffer, 1, _size);
			break;
        }
			
        case MLActivationFunctionTypeRectifiedLinear: {
            
            // Apply formula: delta[i] = (error[i] < 0.0 ? 0.0 : error[i])
            ML_VSMUL(_errorBuffer, 1, &__one, _deltaBuffer, 1, _size);
            ML_VTHRES(_errorBuffer, 1, &__zero, _errorBuffer, 1, _size);
            break;
        }

        case MLActivationFunctionTypeStep: {
			if (self.nextLayer)
				@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Can't backpropagate in a hidden layer with step function"
																		 userInfo:@{@"layer": [NSNumber numberWithUnsignedInteger:self.index]}];
			
			// Apply formula: delta[i] = error[i]
			ML_VSMUL(_errorBuffer, 1, &__one, _deltaBuffer, 1, _size);
			break;
        }
			
        case MLActivationFunctionTypeSigmoid: {
			switch (costType) {
                case MLCostFunctionTypeCrossEntropy: {
					
					// Apply formula: delta[i] = error[i]
					ML_VSMUL(_errorBuffer, 1, &__one, _deltaBuffer, 1, _size);
					break;
                }
					
                case MLCostFunctionTypeSquaredError: {
                    MLReal *tempBuffer= MLAllocRealBuffer(self.size);
			
					// Apply formula: delta[i] = output[i] * (1 - output[i]) * error[i]
					ML_VSMUL(_outputBuffer, 1, &__minusOne, tempBuffer, 1, _size);
					ML_VSADD(tempBuffer, 1, &__one, tempBuffer, 1, _size);
					ML_VMUL(tempBuffer, 1, _outputBuffer, 1, tempBuffer, 1, _size);
					ML_VMUL(tempBuffer, 1, _errorBuffer, 1, _deltaBuffer, 1, _size);
                    
                    MLFreeRealBuffer(tempBuffer);
					break;
                }
			}
			break;
        }
			
        case MLActivationFunctionTypeTanH: {
            MLReal *tempBuffer= MLAllocRealBuffer(self.size);
			
			// Apply formula: delta[i] = (1 - (output[i] * output[i])) * error[i]
			ML_VSQ(_outputBuffer, 1, tempBuffer, 1, _size);
			ML_VSMUL(tempBuffer, 1, &__minusOne, tempBuffer, 1, _size);
			ML_VSADD(tempBuffer, 1, &__one, tempBuffer, 1, _size);
			ML_VMUL(tempBuffer, 1, _errorBuffer, 1, _deltaBuffer, 1, _size);
            
            MLFreeRealBuffer(tempBuffer);
			break;
        }
	}
	
	// Second step: compute new weights for each neuron
    for (MLNeuron *neuron in _neurons)
        [neuron backPropagateWithAlgorithm:backPropType learningRate:learningRate delta:_deltaBuffer[neuron.index]];
}

- (void) updateWeights {
	if (!_neurons)
		@throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Neuron layer not yet set up"
																 userInfo:@{@"layer": [NSNumber numberWithUnsignedInteger:self.index]}];
	
	// Second step: update weights for each neuron
    for (MLNeuron *neuron in _neurons)
		[neuron updateWeights];
}


#pragma mark -
#pragma mark Properties

@synthesize funcType= _funcType;

@synthesize errorBuffer= _errorBuffer;
@synthesize deltaBuffer= _deltaBuffer;

@synthesize outputBuffer= _outputBuffer;

@synthesize usingBias= _usingBias;
@synthesize neurons= _neurons;


@end

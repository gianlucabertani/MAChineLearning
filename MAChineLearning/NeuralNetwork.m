//
//  NeuralNetwork.m
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

#import "NeuralNetwork.h"
#import "InputLayer.h"
#import "NeuronLayer.h"
#import "Neuron.h"
#import "NeuralNetworkException.h"

#import <Accelerate/Accelerate.h>

#define NEURAL_NET_MEMORY_ALIGNMENT          (128)

#define CONFIG_PARAM_LAYER_SIZES             (@"layerSizes")
#define CONFIG_PARAM_OUTPUT_FUNCTION_TYPE    (@"outputFunctionType")
#define CONFIG_PARAM_LAYER                   (@"layer%d")
#define CONFIG_PARAM_WEIGHTS                 (@"weights")
#define CONFIG_PARAM_BIAS                    (@"bias")


#pragma mark -
#pragma mark NeuralNetwork extension

@interface NeuralNetwork () {
	NSMutableArray *_layers;
	ActivationFunctionType _funcType;
	
	int _inputSize;
	nnREAL *_inputBuffer;
	
	int _outputSize;
	nnREAL *_outputBuffer;
	nnREAL *_expectedOutputBuffer;
	nnREAL *_errorBuffer;
	
	NeuralNetworkStatus _status;
}


@end


#pragma mark -
#pragma mark NeuralNetwork implementations

@implementation NeuralNetwork


#pragma mark -
#pragma mark Initialization

+ (NeuralNetwork *) createNetworkFromConfigurationDictionary:(NSDictionary *)config {
	
	// Get sizes and function from configuration
	NSArray *sizes= [config objectForKey:CONFIG_PARAM_LAYER_SIZES];
	if (!sizes)
		@throw [NeuralNetworkException neuralNetworkExceptionWithReason:@"Invalid configuration (missing layer sizes)"
															   userInfo:@{@"config": config}];
	
	NSNumber *funcType= [config objectForKey:CONFIG_PARAM_OUTPUT_FUNCTION_TYPE];
	if (!funcType)
		@throw [NeuralNetworkException neuralNetworkExceptionWithReason:@"Invalid configuration (missing output function type)"
															   userInfo:@{@"config": config}];

	// Create the network
	NeuralNetwork *network= [[NeuralNetwork alloc] initWithLayerSizes:sizes outputFunctionType:[funcType intValue]];
	
	// Get weights from configuration
	for (int i= 1; i < [network.layers count]; i++) {
		NeuronLayer *neuronLayer= [network.layers objectAtIndex:i];
		Layer *previousLayer= neuronLayer.previousLayer;
		
		NSString *layerParam= [NSString stringWithFormat:CONFIG_PARAM_LAYER, i];
		NSArray *layerConfig= [config objectForKey:layerParam];
		if (!layerConfig)
			@throw [NeuralNetworkException neuralNetworkExceptionWithReason:@"Invalid configuration (missing layer configuration)"
																   userInfo:@{@"config": config,
																			  @"layer": [NSNumber numberWithInt:i]}];

		for (int j= 0; j < neuronLayer.size; j++) {
			Neuron *neuron= [neuronLayer.neurons objectAtIndex:j];

			NSDictionary *neuronConfig= [layerConfig objectAtIndex:j];
			if (!neuronConfig)
				@throw [NeuralNetworkException neuralNetworkExceptionWithReason:@"Invalid configuration (missing neuron configuration)"
																	   userInfo:@{@"config": config,
																				  @"layer": [NSNumber numberWithInt:i],
																				  @"neuron": [NSNumber numberWithInt:j]}];
			
			NSNumber *bias= [neuronConfig objectForKey:CONFIG_PARAM_BIAS];
			if (!bias)
				@throw [NeuralNetworkException neuralNetworkExceptionWithReason:@"Invalid configuration (missing neuron bias)"
																	   userInfo:@{@"config": config,
																				  @"layer": [NSNumber numberWithInt:i],
																				  @"neuron": [NSNumber numberWithInt:j]}];

			neuronLayer.biasBuffer[neuron.index]= [bias doubleValue];
			
			NSArray *weights= [neuronConfig objectForKey:CONFIG_PARAM_WEIGHTS];
			if (!weights)
				@throw [NeuralNetworkException neuralNetworkExceptionWithReason:@"Invalid configuration (missing neuron weights list)"
																	   userInfo:@{@"config": config,
																				  @"layer": [NSNumber numberWithInt:i],
																				  @"neuron": [NSNumber numberWithInt:j]}];
			
			for (int k= 0; k < previousLayer.size; k++) {
				NSNumber *weight= [weights objectAtIndex:k];
				if (!weight)
					@throw [NeuralNetworkException neuralNetworkExceptionWithReason:@"Invalid configuration (missing neuron weight)"
																		   userInfo:@{@"config": config,
																					  @"layer": [NSNumber numberWithInt:i],
																					  @"neuron": [NSNumber numberWithInt:j],
																					  @"weight": [NSNumber numberWithInt:k]}];
				
				neuron.weights[k]= [weight doubleValue];
			}
		}
	}
	
	return network;
}

+ (NeuralNetwork *) createNetworkWithLayerSizes:(NSArray *)sizes outputFunctionType:(ActivationFunctionType)funcType {
	NeuralNetwork *network= [[NeuralNetwork alloc] initWithLayerSizes:sizes outputFunctionType:funcType];
	
	return network;
}

- (id) initWithLayerSizes:(NSArray *)sizes outputFunctionType:(ActivationFunctionType)funcType {
	if ((self = [super init])) {
		
		// Initialize the layers: layer 0 is the input layer (we use
		// a neuron layer just for practicity), while the last layer
		// is the output layer
		_layers= [NSMutableArray array];
		_funcType= funcType;

		int i= 0;
		for (NSNumber *size in sizes) {
			if (![size isKindOfClass:[NSNumber class]])
				@throw [NeuralNetworkException neuralNetworkExceptionWithReason:@"Invalid size specified"
																	   userInfo:@{@"size": size}];
			if (i == 0) {
				
				// Create input layer
				InputLayer *layer= [[InputLayer alloc] initWithIndex:i size:[size intValue]];
				[_layers addObject:layer];
			
			} else if (i == [sizes count] -1) {
				
				// Create output neuron layer
				NeuronLayer *layer= [[NeuronLayer alloc] initWithIndex:i size:[size intValue] activationFunctionType:funcType];
				[_layers addObject:layer];
			
			} else {
				
				// Create hidden neuron layer
				NeuronLayer *layer= [[NeuronLayer alloc] initWithIndex:i size:[size intValue] activationFunctionType:ActivationFunctionTypeLogistic];
				[_layers addObject:layer];
			}
			
			i++;
		}
		
		for (int i= 0; i < [_layers count]; i++){
			Layer *layer= [_layers objectAtIndex:i];
			
			// Setup layer relationships
			layer.previousLayer= (i > 0) ? [_layers objectAtIndex:i -1] : nil;
			layer.nextLayer= (i < [_layers count] -1) ? [_layers objectAtIndex:i +1] : nil;
			
			// Setup neurons, and input and output buffer pointers
			if (i == 0) {
				_inputSize= layer.size;
				_inputBuffer= [(InputLayer *) layer inputBuffer];
			
			} else if (i == [_layers count] -1) {
				[(NeuronLayer *) layer setUp];
				
				_outputSize= layer.size;
				_outputBuffer= [(NeuronLayer *) layer outputBuffer];
				_errorBuffer= [(NeuronLayer *) layer errorBuffer];

			} else
				[(NeuronLayer *) layer setUp];
		}
		
		int err= posix_memalign((void **) &_expectedOutputBuffer,
								NEURAL_NET_MEMORY_ALIGNMENT,
								sizeof(nnREAL) * _outputSize);
		if (err)
			@throw [NeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating buffer"
																   userInfo:@{@"buffer": @"expectedOutputBuffer",
																			  @"error": [NSNumber numberWithInt:err]}];
		
		_status= NeuralNetworkStatusIdle;
	}
	
	return self;
}

- (void) dealloc {
	
	// Deallocate buffers
	free(_expectedOutputBuffer);
	_expectedOutputBuffer= NULL;
}


#pragma mark -
#pragma mark Operations

- (void) feedForward {
	_status= NeuralNetworkStatusFeededForward;
	
	// Apply forward propagation
	for (int i= 1; i < [_layers count]; i++) {
		NeuronLayer *layer= [_layers objectAtIndex:i];
		
		[layer feedForward];
	}
}

- (void) backPropagateWithLearningRate:(nnREAL)learningRate {
	
	// Check call sequence
	switch (_status) {
		case NeuralNetworkStatusFeededForward:
			break;
			
		default:
			@throw [NeuralNetworkException neuralNetworkExceptionWithReason:@"Wrong call sequence: network must be feeded forward before it can be back propagated"
																   userInfo:@{@"status": [NSNumber numberWithInt:_status]}];
	}
	
	_status= NeuralNetworkStatusBackPropagated;
	
	// Apply backward propagation
	for (int i= (int) [_layers count] -1; i > 0; i--) {
		NeuronLayer *layer= [_layers objectAtIndex:i];
		
		if (i == [_layers count] -1) {
			
			// Error on output layer is the difference between expected and actual output;
			// NOTE: operands are inverted compared to documentation (see function
			// definition for nnREAL order)
			nnVDSP_VSUB(_outputBuffer, 1, _expectedOutputBuffer, 1, _errorBuffer, 1, _outputSize);
			
		} else
			[layer fetchErrorFromNextLayer];
		
		[layer backPropagateWithLearningRate:learningRate];
	}
}

- (void) updateWeights {
	
	// Check call sequence
	switch (_status) {
		case NeuralNetworkStatusBackPropagated:
			break;
			
		default:
			@throw [NeuralNetworkException neuralNetworkExceptionWithReason:@"Wrong call sequence: network must be back propagated before weights can be updated"
																   userInfo:@{@"status": [NSNumber numberWithInt:_status]}];
	}
	
	_status= NeuralNetworkStatusWeightsUpdated;

	// Apply new weights
	for (int i= 1; i < [_layers count]; i++) {
		NeuronLayer *layer= [_layers objectAtIndex:i];
		
		[layer updateWeights];
	}
}

- (void) terminate {
	_inputSize= 0;
	_inputBuffer= NULL;
	
	_outputSize= 0;
	_outputBuffer= NULL;
	_errorBuffer= NULL;
	
	[_layers removeAllObjects];
	_layers= nil;
}


#pragma mark -
#pragma mark Configuration load/save

- (NSDictionary *) saveConfigurationToDictionary {
	NSMutableDictionary *config= [[NSMutableDictionary alloc] initWithCapacity:[_layers count] +1];
	
	// Save the activation function
	[config setObject:[NSNumber numberWithInt:_funcType] forKey:CONFIG_PARAM_OUTPUT_FUNCTION_TYPE];

	// Save layer sizes
	NSMutableArray *sizes= [[NSMutableArray alloc] initWithCapacity:[_layers count]];
	for (Layer *layer in _layers)
		[sizes addObject:[NSNumber numberWithInt:layer.size]];
	
	[config setObject:sizes forKey:CONFIG_PARAM_LAYER_SIZES];
	
	// Save weights for each non-input layer
	for (int i= 1; i < [_layers count]; i++) {
		NeuronLayer *neuronLayer= [_layers objectAtIndex:i];
		Layer *previousLayer= neuronLayer.previousLayer;
		
		NSMutableArray *layerConfig= [[NSMutableArray alloc] initWithCapacity:neuronLayer.size];
		for (int j= 0; j < neuronLayer.size; j++) {
			Neuron *neuron= [neuronLayer.neurons objectAtIndex:j];
			
			NSMutableArray *weights= [[NSMutableArray alloc] initWithCapacity:previousLayer.size];
			for (int k= 0; k < previousLayer.size; k++)
				[weights addObject:[NSNumber numberWithDouble:neuron.weights[k]]];
			
			NSMutableDictionary *neuronConfig= [[NSMutableDictionary alloc] initWithCapacity:2];
			[neuronConfig setObject:weights forKey:CONFIG_PARAM_WEIGHTS];
			[neuronConfig setObject:[NSNumber numberWithDouble:neuronLayer.biasBuffer[neuron.index]] forKey:CONFIG_PARAM_BIAS];
			
			[layerConfig addObject:neuronConfig];
		}
		
		NSString *layerParam= [NSString stringWithFormat:CONFIG_PARAM_LAYER, i];
		[config setObject:layerConfig forKey:layerParam];
	}
	
	return config;
}


#pragma mark -
#pragma mark Properties

@synthesize layers= _layers;

@synthesize inputSize= _inputSize;
@synthesize inputBuffer= _inputBuffer;

@synthesize outputSize= _outputSize;
@synthesize outputBuffer= _outputBuffer;
@synthesize expectedOutputBuffer= _expectedOutputBuffer;

@synthesize status= _status;


@end

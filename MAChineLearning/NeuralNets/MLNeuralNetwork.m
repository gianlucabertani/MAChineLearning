//
//  MLNeuralNetwork.m
//  MAChineLearning
//
//  Created by Gianluca Bertani on 01/03/15.
//  Copyright (c) 2015-2018 Gianluca Bertani. All rights reserved.
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

#import "MLNeuralNetwork.h"
#import "MLInputLayer.h"
#import "MLNeuronLayer.h"
#import "MLNeuron.h"
#import "MLNeuralNetworkException.h"

#import "MLAlloc.h"


#define CONFIG_PARAM_LAYER_SIZES             (@"layerSizes")
#define CONFIG_PARAM_USE_BIAS                (@"useBias")
#define CONFIG_PARAM_COST_FUNCTION_TYPE      (@"costType")
#define CONFIG_PARAM_BACK_PROPAGATION_TYPE   (@"backPropagationType")
#define CONFIG_PARAM_HIDDEN_FUNCTION_TYPE    (@"hiddenFunctionType")
#define CONFIG_PARAM_OUTPUT_FUNCTION_TYPE    (@"outputFunctionType")
#define CONFIG_PARAM_LAYER                   (@"layer%d")
#define CONFIG_PARAM_WEIGHTS                 (@"weights")


#pragma mark -
#pragma mark NeuralNetwork extension

@interface MLNeuralNetwork () {
    NSMutableArray<MLLayer *> *_layers;
    BOOL _useBias;
    MLActivationFunctionType _hiddenFuncType;
    MLActivationFunctionType _funcType;
    MLBackPropagationType _backPropType;
    MLCostFunctionType _costType;
    
    NSUInteger _inputSize;
    MLReal *_inputBuffer;
    
    NSUInteger _outputSize;
    MLReal *_outputBuffer;
    MLReal *_expectedOutputBuffer;
    MLReal *_errorBuffer;
    
    MLNeuralNetworkStatus _status;
}


@end


#pragma mark -
#pragma mark Static constants

static const MLReal __minusOne=              -1.0;
static const MLReal __one=                    1.0;


#pragma mark -
#pragma mark NeuralNetwork implementations

@implementation MLNeuralNetwork


#pragma mark -
#pragma mark Initialization

+ (MLNeuralNetwork *) createNetworkFromConfigurationDictionary:(NSDictionary<NSString *, id> *)config {
    
    // Get sizes and function from configuration
    NSArray<NSNumber *> *sizes= config[CONFIG_PARAM_LAYER_SIZES];
    if (!sizes)
        @throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Invalid configuration: missing layer sizes"
                                                                 userInfo:@{@"config": config}];
    
    // Get use of bias from configuration
    NSNumber *useBias= config[CONFIG_PARAM_USE_BIAS];
    if (!useBias)
        @throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Invalid configuration: missing flag for use of bias"
                                                                 userInfo:@{@"config": config}];
    
    // Get hidden activation function type from configuration
    NSNumber *hiddenFuncType= config[CONFIG_PARAM_HIDDEN_FUNCTION_TYPE];
    if (!hiddenFuncType)
        @throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Invalid configuration: missing hidden function type"
                                                                 userInfo:@{@"config": config}];
    
    NSNumber *funcType= config[CONFIG_PARAM_OUTPUT_FUNCTION_TYPE];
    if (!funcType)
        @throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Invalid configuration: missing output function type"
                                                                 userInfo:@{@"config": config}];
    
    // Get backpropagation type from configuration
    NSNumber *backPropType= config[CONFIG_PARAM_BACK_PROPAGATION_TYPE];
    if (!backPropType)
        @throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Invalid configuration: missing backpropagation type"
                                                                 userInfo:@{@"config": config}];
    
    // Get backpropagation type from configuration
    NSNumber *costType= config[CONFIG_PARAM_COST_FUNCTION_TYPE];
    if (!costType)
        @throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Invalid configuration: missing cost function type"
                                                                 userInfo:@{@"config": config}];
    
    // Create the network
    MLNeuralNetwork *network= [[MLNeuralNetwork alloc] initWithLayerSizes:sizes
                                                                  useBias:useBias.boolValue
                                                         costFunctionType:costType.intValue
                                                      backPropagationType:backPropType.intValue
                                                       hiddenFunctionType:hiddenFuncType.intValue
                                                       outputFunctionType:funcType.intValue];
    
    // Get weights from configuration
    for (int i= 1; i < network.layers.count; i++) {
        MLNeuronLayer *neuronLayer= (MLNeuronLayer *) network.layers[i];
        MLLayer *previousLayer= neuronLayer.previousLayer;
        
        NSString *layerParam= [NSString stringWithFormat:CONFIG_PARAM_LAYER, i];
        NSArray<NSDictionary<NSString *, id> *> *layerConfig= config[layerParam];
        if (!layerConfig)
            @throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Invalid configuration: missing layer configuration"
                                                                     userInfo:@{@"config": config,
                                                                                @"layer": @(i)}];

        for (int j= 0; j < neuronLayer.size; j++) {
            MLNeuron *neuron= neuronLayer.neurons[j];

            NSDictionary<NSString *, id> *neuronConfig= layerConfig[j];
            if (!neuronConfig)
                @throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Invalid configuration: missing neuron configuration"
                                                                         userInfo:@{@"config": config,
                                                                                    @"layer": @(i),
                                                                                    @"neuron": @(j)}];
            
            NSArray<NSNumber *> *weights= neuronConfig[CONFIG_PARAM_WEIGHTS];
            if (!weights)
                @throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Invalid configuration: missing neuron weights list"
                                                                         userInfo:@{@"config": config,
                                                                                    @"layer": @(i),
                                                                                    @"neuron": @(j)}];
            
            for (int k= 0; k < previousLayer.size; k++) {
                NSNumber *weight= weights[k];
                if (!weight)
                    @throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Invalid configuration: missing neuron weight"
                                                                             userInfo:@{@"config": config,
                                                                                        @"layer": @(i),
                                                                                        @"neuron": @(j),
                                                                                        @"weight": @(k)}];
                
                neuron.weights[k]= weight.doubleValue;
            }
        }
    }
    
    return network;
}

+ (MLNeuralNetwork *) createNetworkWithLayerSizes:(NSArray<NSNumber *> *)sizes
                               outputFunctionType:(MLActivationFunctionType)funcType {
    
    MLNeuralNetwork *network= [[MLNeuralNetwork alloc] initWithLayerSizes:sizes
                                                                  useBias:YES
                                                         costFunctionType:MLCostFunctionTypeSquaredError
                                                      backPropagationType:MLBackPropagationTypeResilient
                                                       hiddenFunctionType:MLActivationFunctionTypeSigmoid
                                                       outputFunctionType:funcType];
    
    return network;
}

+ (MLNeuralNetwork *) createNetworkWithLayerSizes:(NSArray<NSNumber *> *)sizes
                                 costFunctionType:(MLCostFunctionType)costType
                               outputFunctionType:(MLActivationFunctionType)funcType {
    
    MLNeuralNetwork *network= [[MLNeuralNetwork alloc] initWithLayerSizes:sizes
                                                                  useBias:YES
                                                         costFunctionType:costType
                                                      backPropagationType:MLBackPropagationTypeResilient
                                                       hiddenFunctionType:MLActivationFunctionTypeSigmoid
                                                       outputFunctionType:funcType];
    
    return network;
}

+ (MLNeuralNetwork *) createNetworkWithLayerSizes:(NSArray<NSNumber *> *)sizes
                              backPropagationType:(MLBackPropagationType)backPropType
                               outputFunctionType:(MLActivationFunctionType)funcType {
    
    MLNeuralNetwork *network= [[MLNeuralNetwork alloc] initWithLayerSizes:sizes
                                                                  useBias:YES
                                                         costFunctionType:MLCostFunctionTypeSquaredError
                                                      backPropagationType:backPropType
                                                       hiddenFunctionType:MLActivationFunctionTypeSigmoid
                                                       outputFunctionType:funcType];
    
    return network;
}

+ (MLNeuralNetwork *) createNetworkWithLayerSizes:(NSArray<NSNumber *> *)sizes
                                 costFunctionType:(MLCostFunctionType)costType
                              backPropagationType:(MLBackPropagationType)backPropType
                               outputFunctionType:(MLActivationFunctionType)funcType {
    
    MLNeuralNetwork *network= [[MLNeuralNetwork alloc] initWithLayerSizes:sizes
                                                                  useBias:YES
                                                         costFunctionType:costType
                                                      backPropagationType:backPropType
                                                       hiddenFunctionType:MLActivationFunctionTypeSigmoid
                                                       outputFunctionType:funcType];
    
    return network;
}

+ (MLNeuralNetwork *) createNetworkWithLayerSizes:(NSArray<NSNumber *> *)sizes
                              backPropagationType:(MLBackPropagationType)backPropType
                               hiddenFunctionType:(MLActivationFunctionType)hiddenFuncType
                               outputFunctionType:(MLActivationFunctionType)funcType {
    
    MLNeuralNetwork *network= [[MLNeuralNetwork alloc] initWithLayerSizes:sizes
                                                                  useBias:YES
                                                         costFunctionType:MLCostFunctionTypeSquaredError
                                                      backPropagationType:backPropType
                                                       hiddenFunctionType:hiddenFuncType
                                                       outputFunctionType:funcType];
    
    return network;
}

+ (MLNeuralNetwork *) createNetworkWithLayerSizes:(NSArray<NSNumber *> *)sizes
                                 costFunctionType:(MLCostFunctionType)costType
                              backPropagationType:(MLBackPropagationType)backPropType
                               hiddenFunctionType:(MLActivationFunctionType)hiddenFuncType
                               outputFunctionType:(MLActivationFunctionType)funcType {
    
    MLNeuralNetwork *network= [[MLNeuralNetwork alloc] initWithLayerSizes:sizes
                                                                  useBias:YES
                                                         costFunctionType:costType
                                                      backPropagationType:backPropType
                                                       hiddenFunctionType:hiddenFuncType
                                                       outputFunctionType:funcType];
    
    return network;
}

- (instancetype) initWithLayerSizes:(NSArray<NSNumber *> *)sizes
                            useBias:(BOOL)useBias
                   costFunctionType:(MLCostFunctionType)costType
                backPropagationType:(MLBackPropagationType)backPropType
                 hiddenFunctionType:(MLActivationFunctionType)hiddenFuncType
                 outputFunctionType:(MLActivationFunctionType)funcType {
    
    if ((self = [super init])) {
        
        // Checks
        switch (costType) {
            case MLCostFunctionTypeCrossEntropy: {
                if ((hiddenFuncType != MLActivationFunctionTypeSigmoid) || (funcType != MLActivationFunctionTypeSigmoid))
                    @throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Wrong cost function: cross entropy can be used only with sigmoid activation function for all layers"
                                                                             userInfo:@{@"hiddenFunctionType": @(hiddenFuncType),
                                                                                        @"outputFunctionType": @(funcType)}];
                break;
            }
                
            default:
                break;
        }
        
        // Initialize the layers: layer 0 is the input layer,
        // while the last layer is the output layer
        _layers= [NSMutableArray array];
        _useBias= useBias;
        _backPropType= backPropType;
        _hiddenFuncType= hiddenFuncType;
        _funcType= funcType;
        _costType= costType;

        int i= 0;
        for (NSNumber *size in sizes) {
            if (![size isKindOfClass:[NSNumber class]])
                @throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Invalid size specified"
                                                                         userInfo:@{@"size": size}];

            if (i == 0) {
                
                // Create input layer
                MLInputLayer *layer= [[MLInputLayer alloc] initWithIndex:i size:size.intValue];
                [_layers addObject:layer];
            
            } else if (i == sizes.count -1) {
                
                // Create output neuron layer
                MLNeuronLayer *layer= [[MLNeuronLayer alloc] initWithIndex:i size:size.intValue useBias:NO activationFunctionType:funcType];
                [_layers addObject:layer];
            
            } else {
                
                // Create hidden neuron layer
                MLNeuronLayer *layer= [[MLNeuronLayer alloc] initWithIndex:i size:size.intValue useBias:useBias activationFunctionType:hiddenFuncType];
                [_layers addObject:layer];
            }
            
            i++;
        }
        
        // Layers setup: create neurons for each layer
        i= 0;
        for (MLLayer *layer in _layers) {
            
            // Setup layer relationships
            layer.previousLayer= (i > 0) ? _layers[i -1] : nil;
            layer.nextLayer= (i < _layers.count -1) ? _layers[i +1] : nil;
            
            // Setup neurons, and input and output buffer pointers
            [layer setUp];
            
            if (i == 0) {
                _inputSize= layer.size;
                _inputBuffer= ((MLInputLayer *) layer).inputBuffer;
            
            } else if (i == _layers.count -1) {
                _outputSize= layer.size;
                _outputBuffer= ((MLNeuronLayer *) layer).outputBuffer;
                _errorBuffer= ((MLNeuronLayer *) layer).errorBuffer;
            }
            
            i++;
        }
        
        // Neurons setup: during setup each neuron connects its weights
        // pointer for weight gathering during backpropagation, for this
        // reason we have to go from output layers backwords
        for (NSUInteger i= _layers.count -1; i > 0; i--) {
            MLNeuronLayer *neuronLayer= (MLNeuronLayer *) _layers[i];
            for (MLNeuron *neuron in neuronLayer.neurons)
                [neuron setUpForBackpropagationWithAlgorithm:_backPropType];
        }
        
        _expectedOutputBuffer= MLAllocRealBuffer(_outputSize);
        
        _status= MLNeuralNetworkStatusIdle;
    }
    
    return self;
}

- (void) dealloc {
    
    // Deallocate buffer
    MLFreeRealBuffer(_expectedOutputBuffer);
    _expectedOutputBuffer= NULL;
}


#pragma mark -
#pragma mark Randomization

- (void) randomizeWeights {
    
    // Randomize each layer
    for (int i= 1; i < _layers.count; i++) {
        MLNeuronLayer *layer= (MLNeuronLayer *) _layers[i];
        
        [layer randomizeWeights];
    }
}


#pragma mark -
#pragma mark Operations

- (void) feedForward {
    _status= MLNeuralNetworkStatusFeededForward;
    
    // Apply forward propagation
    for (int i= 1; i < _layers.count; i++) {
        MLNeuronLayer *layer= (MLNeuronLayer *) _layers[i];
        
        [layer feedForward];
    }
}

- (void) backPropagate {
    
    // Checks
    switch (_backPropType) {
        case MLBackPropagationTypeStandard:
            @throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Invalid learning rate: standard backpropagation requires a positive learning rate"
                                                                     userInfo:nil];
            
        case MLBackPropagationTypeResilient:
            [self backPropagateWithLearningRate:0.0];
            break;
    }
}

- (void) backPropagateWithLearningRate:(MLReal)learningRate {
    
    // Checks
    switch (_backPropType) {
        case MLBackPropagationTypeStandard:
            if (learningRate <= 0.0)
                @throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Invalid learning rate: standard backpropagation requires a positive learning rate"
                                                                         userInfo:@{@"learningRate": @(learningRate)}];
            break;
            
        case MLBackPropagationTypeResilient:
            if (learningRate != 0.0)
                @throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Invalid learning rate: resilient backpropagation makes no use of learning rate, should not be passed"
                                                                         userInfo:nil];
            break;
    }

    // Check call sequence
    switch (_status) {
        case MLNeuralNetworkStatusFeededForward:
            break;
            
        default:
            @throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Wrong call sequence: network must be feeded forward before it can be back propagated"
                                                                     userInfo:@{@"status": @(_status)}];
    }
    
    _status= MLNeuralNetworkStatusBackPropagated;
    
    // Apply backward propagation
    for (NSUInteger i= _layers.count -1; i > 0; i--) {
        MLNeuronLayer *layer= (MLNeuronLayer *) _layers[i];
        
        if (i == _layers.count -1) {
            
            // Error on output layer is the difference between expected and actual output
            ML_VSUB(_outputBuffer, 1, _expectedOutputBuffer, 1, _errorBuffer, 1, _outputSize);
            
        } else
            [layer fetchErrorFromNextLayer];
        
        [layer backPropagateWithAlgorithm:_backPropType learningRate:learningRate costFunction:_costType];
    }
}

- (void) updateWeights {
    
    // Check call sequence
    switch (_status) {
        case MLNeuralNetworkStatusBackPropagated:
            break;
            
        default:
            @throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"Wrong call sequence: network must be back propagated before weights can be updated"
                                                                     userInfo:@{@"status": @(_status)}];
    }
    
    _status= MLNeuralNetworkStatusWeightsUpdated;

    // Apply new weights
    for (int i= 1; i < _layers.count; i++) {
        MLNeuronLayer *layer= (MLNeuronLayer *) _layers[i];
        
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

- (NSDictionary<NSString *, id> *) saveConfigurationToDictionary {
    NSMutableDictionary<NSString *, id> *config= [[NSMutableDictionary alloc] initWithCapacity:_layers.count +1];
    
    // Save the basic configuration
    config[CONFIG_PARAM_COST_FUNCTION_TYPE]= @(_costType);
    config[CONFIG_PARAM_OUTPUT_FUNCTION_TYPE]= @(_funcType);
    config[CONFIG_PARAM_BACK_PROPAGATION_TYPE]= @(_backPropType);
    config[CONFIG_PARAM_HIDDEN_FUNCTION_TYPE]= @(_hiddenFuncType);
    config[CONFIG_PARAM_USE_BIAS]= @(_useBias);

    // Save layer sizes
    NSMutableArray<NSNumber *> *sizes= [[NSMutableArray alloc] initWithCapacity:_layers.count];
    for (MLLayer *layer in _layers) {
        BOOL hasBiasNeuron= ([layer isKindOfClass:[MLNeuronLayer class]] && ((MLNeuronLayer *) layer).usingBias);
        [sizes addObject:@(layer.size - (hasBiasNeuron ? 1 : 0))];
    }
    
    config[CONFIG_PARAM_LAYER_SIZES]= sizes;
    
    // Save weights for each non-input layer
    for (int i= 1; i < _layers.count; i++) {
        MLNeuronLayer *neuronLayer= (MLNeuronLayer *) _layers[i];
        MLLayer *previousLayer= neuronLayer.previousLayer;
        
        NSMutableArray<NSDictionary<NSString *, id> *> *layerConfig= [[NSMutableArray alloc] initWithCapacity:neuronLayer.size];
        for (int j= 0; j < neuronLayer.size; j++) {
            MLNeuron *neuron= neuronLayer.neurons[j];
            
            NSMutableArray<NSNumber *> *weights= [[NSMutableArray alloc] initWithCapacity:previousLayer.size];
            for (int k= 0; k < previousLayer.size; k++)
                [weights addObject:@(neuron.weights[k])];
            
            NSMutableDictionary<NSString *, id> *neuronConfig= [[NSMutableDictionary alloc] initWithCapacity:1];
            neuronConfig[CONFIG_PARAM_WEIGHTS]= weights;
            
            [layerConfig addObject:neuronConfig];
        }
        
        NSString *layerParam= [NSString stringWithFormat:CONFIG_PARAM_LAYER, i];
        config[layerParam]= layerConfig;
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

@dynamic cost;

- (MLReal) cost {
    MLReal cost= 0.0;
    
    switch (_costType) {
        case MLCostFunctionTypeSquaredError: {
            
            // Apply formula: cost = 0.5 * Sum((expectedOutput[i] - output[i])^2)
            ML_VSUB(_outputBuffer, 1, _expectedOutputBuffer, 1, _errorBuffer, 1, _outputSize);
            ML_SVESQ(_errorBuffer, 1, &cost, _outputSize);
            cost *= 0.5;
            break;
        }
            
        case MLCostFunctionTypeCrossEntropy: {
            MLReal *tempBuffer= MLAllocRealBuffer(_outputSize);
            
            // An "int" size is needed by vvlog,
            // the others still use _outputSize
            int outputSize= (int) _outputSize;

            // Apply formula: cost = -Sum(expectedOutput[i] * ln(output[i]) + (1 - expectedOutput[i]) * ln(1 - output[i]))
            ML_VSMUL(_outputBuffer, 1, &__minusOne, tempBuffer, 1, _outputSize);
            ML_VSADD(tempBuffer, 1, &__one, tempBuffer, 1, _outputSize);
            ML_VVLOG(_errorBuffer, tempBuffer, &outputSize);
            
            ML_VMUL(_errorBuffer, 1, _expectedOutputBuffer, 1, tempBuffer, 1, _outputSize);
            ML_VSMUL(tempBuffer, 1, &__minusOne, tempBuffer, 1, _outputSize);
            ML_VADD(tempBuffer, 1, _errorBuffer, 1, _errorBuffer, 1, _outputSize);
            
            ML_VVLOG(tempBuffer, _outputBuffer, &outputSize);
            ML_VMUL(tempBuffer, 1, _expectedOutputBuffer, 1, tempBuffer, 1, _outputSize);
            ML_VADD(tempBuffer, 1, _errorBuffer, 1, _errorBuffer, 1, _outputSize);
            
            ML_SVE(_errorBuffer, 1, &cost, _outputSize);
            cost *= -1.0;
            
            MLFreeRealBuffer(tempBuffer);
            break;
        }
    }
    
    return cost;
}

@synthesize status= _status;


@end

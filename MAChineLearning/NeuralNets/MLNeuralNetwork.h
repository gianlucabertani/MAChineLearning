//
//  MLNeuralNetwork.h
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

#import <Foundation/Foundation.h>

#import "MLReal.h"

#import "MLNeuralNetworkStatus.h"
#import "MLBackPropagationType.h"
#import "MLActivationFunctionType.h"


@interface MLNeuralNetwork : NSObject


#pragma mark -
#pragma mark Initialization

+ (MLNeuralNetwork *) createNetworkFromConfigurationDictionary:(NSDictionary *)config;
+ (MLNeuralNetwork *) createNetworkWithLayerSizes:(NSArray *)sizes backPropagationType:(MLBackPropagationType)backPropType outputFunctionType:(MLActivationFunctionType)funcType;

- (instancetype) initWithLayerSizes:(NSArray *)sizes backPropagationType:(MLBackPropagationType)backPropType outputFunctionType:(MLActivationFunctionType)funcType;


#pragma mark -
#pragma mark Randomization

- (void) randomizeWeights;


#pragma mark -
#pragma mark Operations

- (void) feedForward;
- (void) backPropagate;
- (void) backPropagateWithLearningRate:(MLReal)learningRate;
- (void) updateWeights;

- (void) terminate;


#pragma mark -
#pragma mark Configuration

- (NSDictionary *) saveConfigurationToDictionary;


#pragma mark -
#pragma mark Properties

@property (nonatomic, readonly) NSArray *layers;

@property (nonatomic, readonly) NSUInteger inputSize;
@property (nonatomic, readonly) MLReal *inputBuffer;

@property (nonatomic, readonly) NSUInteger outputSize;
@property (nonatomic, readonly) MLReal *outputBuffer;
@property (nonatomic, readonly) MLReal *expectedOutputBuffer;

@property (nonatomic, readonly) MLNeuralNetworkStatus status;


@end

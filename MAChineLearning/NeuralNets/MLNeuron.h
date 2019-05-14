//
//  MLNeuron.h
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

#import <Foundation/Foundation.h>

#import "MLReal.h"

#import "MLBackPropagationType.h"
#import "MLActivationFunctionType.h"


@class MLNeuronLayer;

@interface MLNeuron : NSObject


#pragma mark -
#pragma mark Initialization

- (nonnull instancetype) init NS_UNAVAILABLE;

- (nonnull instancetype) initWithLayer:(nonnull MLNeuronLayer *)layer
                                 index:(NSUInteger)index
                          outputBuffer:(nonnull MLReal *)outputBuffer
                             inputSize:(NSUInteger)inputSize
                           inputBuffer:(nonnull MLReal *)inputBuffer
                                       NS_DESIGNATED_INITIALIZER;


#pragma mark -
#pragma mark Setup and randomization

- (void) setUpForBackpropagationWithAlgorithm:(MLBackPropagationType)backPropType;
- (void) randomizeWeightsWithBeta:(MLReal)beta;


#pragma mark -
#pragma mark Operations

- (void) feedForward;

- (void) backPropagateWithAlgorithm:(MLBackPropagationType)backPropType
                       learningRate:(MLReal)learningRate
                              delta:(MLReal)delta;

- (void) updateWeights;


#pragma mark -
#pragma mark Properties

@property (nonatomic, readonly, nonnull) MLNeuronLayer *layer;

@property (nonatomic, readonly) NSUInteger index;
@property (nonatomic, readonly, nonnull) MLReal *outputBuffer;

@property (nonatomic, readonly) NSUInteger inputSize;
@property (nonatomic, readonly, nonnull) MLReal *inputBuffer;

@property (nonatomic, readonly, nonnull) MLReal *weights;
@property (nonatomic, readonly, nonnull) MLReal *weightsDelta;

@property (nonatomic, readonly) MLReal * _Nonnull * _Nullable nextLayerWeightPtrs;
@property (nonatomic, readonly) MLReal * _Nonnull * _Nullable nextLayerWeightDeltaPtrs;

@property (nonatomic, readonly) MLReal error;
@property (nonatomic, readonly) MLReal delta;


@end

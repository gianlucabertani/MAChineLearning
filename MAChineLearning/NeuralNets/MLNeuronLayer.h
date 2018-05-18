//
//  MLNeuronLayer.h
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

#import <Foundation/Foundation.h>

#import "MLReal.h"

#import "MLLayer.h"
#import "MLBackPropagationType.h"
#import "MLActivationFunctionType.h"
#import "MLCostFunctionType.h"


@class MLNeuron;

@interface MLNeuronLayer : MLLayer


#pragma mark -
#pragma mark Initialization

- (nonnull instancetype) initWithIndex:(NSUInteger)index
                                  size:(NSUInteger)size
                               useBias:(BOOL)useBias
                activationFunctionType:(MLActivationFunctionType)funcType;


#pragma mark -
#pragma mark Randomization

- (void) randomizeWeights;


#pragma mark -
#pragma mark Operations

- (void) feedForward;

- (void) fetchErrorFromNextLayer;

- (void) backPropagateWithAlgorithm:(MLBackPropagationType)backPropType
                       learningRate:(MLReal)learningRate
                       costFunction:(MLCostFunctionType)costType;

- (void) updateWeights;


#pragma mark -
#pragma mark Properties

@property (nonatomic, readonly) MLActivationFunctionType funcType;

@property (nonatomic, readonly, nonnull) MLReal *errorBuffer;
@property (nonatomic, readonly, nonnull) MLReal *deltaBuffer;

@property (nonatomic, readonly, nonnull) MLReal *outputBuffer;

@property (nonatomic, readonly) BOOL usingBias;
@property (nonatomic, readonly, nonnull) NSArray<MLNeuron *> *neurons;


@end

//
//  MLBiasNeuron.m
//  MAChineLearning
//
//  Created by Gianluca Bertani on 21/06/15.
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

#import "MLBiasNeuron.h"


#pragma mark -
#pragma mark Static constants

static const MLReal __one=                    1.0;


#pragma mark -
#pragma mark BiasNeuron implementation

@implementation MLBiasNeuron


#pragma mark -
#pragma mark Initialization

- (instancetype) initWithLayer:(MLNeuronLayer *)layer index:(NSUInteger)index outputBuffer:(MLReal *)outputBuffer inputSize:(NSUInteger)inputSize inputBuffer:(MLReal *)inputBuffer {
	if ((self = [super initWithLayer:layer index:index outputBuffer:outputBuffer inputSize:inputSize inputBuffer:inputBuffer])) {
		
		// Nothing to do
	}
	
	return self;
}


#pragma mark -
#pragma mark Setup and randomization

- (void) setUpForBackpropagationWithAlgorithm:(MLBackPropagationType)backPropType {
	
	// Use setup for basic backpropagation, to avoid wasting buffers for RPROP
	[super setUpForBackpropagationWithAlgorithm:MLBackPropagationTypeStandard];
}

- (void) randomizeWeights {
	
	// Nothing to do, weights remain 0
}


#pragma mark -
#pragma mark Operations

- (void) feedForward {
	
	// Output is constantly 1
	self.outputBuffer[self.index]= __one;
}

- (void) backPropagateWithAlgorithm:(MLBackPropagationType)backPropType learningRate:(MLReal)learningRate delta:(MLReal)delta {
	
	// Nothing to do, weights remain 0
}

- (void) updateWeights {
	
	// Nothing to do, weights remain 0
}


@end

//
//  InputLayer.m
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

#import "InputLayer.h"
#import "NeuralNetworkException.h"

#import <Accelerate/Accelerate.h>

#define NEURAL_NET_MEMORY_ALIGNMENT          (128)


#pragma mark -
#pragma mark InputLayer extension

@interface InputLayer () {
	nnREAL *_inputBuffer;
}


@end


#pragma mark -
#pragma mark InputLayer implementation

@implementation InputLayer


#pragma mark -
#pragma mark Initialization

- (id) initWithIndex:(int)index size:(int)size {
	if ((self = [super initWithIndex:index size:size])) {
		
		// Allocate buffers
		int err= posix_memalign((void **) &_inputBuffer,
								NEURAL_NET_MEMORY_ALIGNMENT,
								sizeof(nnREAL) * size);
		if (err)
			@throw [NeuralNetworkException neuralNetworkExceptionWithReason:@"Error while allocating buffer"
																   userInfo:@{@"buffer": @"inputBuffer",
																			  @"error": [NSNumber numberWithInt:err]}];
		
		// Clear and fill buffers as needed
		nnVDSP_VCLR(_inputBuffer, 1, size);
	}
	
	return self;
}

- (void) dealloc {
	
	// Deallocate the input buffer
	free(_inputBuffer);
	_inputBuffer= NULL;
}


#pragma mark -
#pragma mark Properties

@synthesize inputBuffer= _inputBuffer;


@end

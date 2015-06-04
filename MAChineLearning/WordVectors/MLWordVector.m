//
//  MLWordVector.m
//  MAChineLearning
//
//  Created by Gianluca Bertani on 03/06/15.
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

#import "MLWordVector.h"
#import "MLWordVectorException.h"

#import "MLReal.h"
#import "MLConstants.h"

#import <Accelerate/Accelerate.h>


#pragma mark -
#pragma mark MLWordVectorMap extension

@interface MLWordVector () {
	MLWordInfo *_wordInfo;
	MLReal *_vector;
	NSUInteger _size;

	BOOL _freeOnDealloc;

	MLReal *_temp1;
	MLReal *_temp2;
	MLReal *_temp3;
}


@end


#pragma mark -
#pragma mark MLWordVectorMap implementation

@implementation MLWordVector


#pragma -
#pragma Initialization

- (instancetype) initWithWordInfo:(MLWordInfo *)wordInfo vector:(MLReal *)vector size:(NSUInteger)size freeVectorOnDealloc:(BOOL)freeOnDealloc {
	if ((self = [super init])) {
		
		// Initialization
		_wordInfo= wordInfo;
		_vector= vector;
		_size= size;
		
		_freeOnDealloc= freeOnDealloc;
		
		int err= posix_memalign((void **) &_temp1,
								BUFFER_MEMORY_ALIGNMENT,
								sizeof(MLReal));
		if (err)
			@throw [MLWordVectorException wordVectorExceptionWithReason:@"Error while allocating buffer"
															   userInfo:@{@"buffer": @"temp1",
																		  @"error": [NSNumber numberWithInt:err]}];
		
		err= posix_memalign((void **) &_temp2,
							BUFFER_MEMORY_ALIGNMENT,
							sizeof(MLReal));
		if (err)
			@throw [MLWordVectorException wordVectorExceptionWithReason:@"Error while allocating buffer"
															   userInfo:@{@"buffer": @"temp2",
																		  @"error": [NSNumber numberWithInt:err]}];
		
		err= posix_memalign((void **) &_temp3,
							BUFFER_MEMORY_ALIGNMENT,
							sizeof(MLReal));
		if (err)
			@throw [MLWordVectorException wordVectorExceptionWithReason:@"Error while allocating buffer"
															   userInfo:@{@"buffer": @"temp3",
																		  @"error": [NSNumber numberWithInt:err]}];
	}
	
	return self;
}

- (void) dealloc {
	if (_freeOnDealloc) {
		free(_vector);
		_vector= NULL;
	}
	
	free(_temp1);
	_temp1= NULL;
	
	free(_temp2);
	_temp2= NULL;
	
	free(_temp3);
	_temp3= NULL;
}


#pragma -
#pragma Vector algebra and comparison

- (MLReal) similarityToVector:(MLWordVector *)vector {
	
	// Checks
	if (_size != vector.size)
		@throw [MLWordVectorException wordVectorExceptionWithReason:@"Vectors must have the same size"
														   userInfo:@{@"size": [NSNumber numberWithUnsignedInteger:_size],
																	  @"vectorSize": [NSNumber numberWithUnsignedInteger:vector.size]}];
	
	// Dot product of vectors
	ML_VDSP_DOTPR(_vector, 1, vector.vector, 1, _temp1, _size);
	
	// Square of magnitude of this vector
	ML_VDSP_SVESQ(_vector, 1, _temp2, _size);
	
	// Square of magnitude of other vector
	ML_VDSP_SVESQ(vector.vector, 1, _temp3, _size);
	
	// Return cosine similarity
	return *_temp1 / ML_SQRT(*_temp2 * *_temp3);
}


#pragma -
#pragma Properties

@synthesize wordInfo= _wordInfo;
@synthesize vector= _vector;
@synthesize size= _size;


@end

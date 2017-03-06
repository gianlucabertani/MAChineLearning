//
//  MLWordVector.m
//  MAChineLearning
//
//  Created by Gianluca Bertani on 03/06/15.
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

#import "MLWordVector.h"
#import "MLWordVectorException.h"

#import "MLReal.h"
#import "MLAlloc.h"


#pragma mark -
#pragma mark MLWordVector extension

@interface MLWordVector () {
	MLReal *_vector;
	NSUInteger _size;

	BOOL _freeOnDealloc;

	MLReal _magnitude;
}


@end


#pragma mark -
#pragma mark MLWordVector implementation

@implementation MLWordVector


#pragma -
#pragma Initialization

- (instancetype) initWithVector:(MLReal *)vector size:(NSUInteger)size freeVectorOnDealloc:(BOOL)freeOnDealloc {
	if ((self = [super init])) {
		
		// Initialization
		_vector= vector;
		_size= size;
		
		_freeOnDealloc= freeOnDealloc;

		// Compute magnitude
		ML_SVESQ(_vector, 1, &_magnitude, _size);
		_magnitude= ML_SQRT(_magnitude);
	}
	
	return self;
}

- (void) dealloc {
	if (_freeOnDealloc) {
		MLFreeRealBuffer(_vector);
		_vector= NULL;
	}
}


#pragma -
#pragma Vector algebra and comparison

- (MLWordVector *) addVector:(MLWordVector *)vector {
	
	// Checks
	if (_size != vector.size)
		@throw [MLWordVectorException wordVectorExceptionWithReason:@"Vectors must have the same size"
														   userInfo:@{@"size": [NSNumber numberWithUnsignedInteger:_size],
																	  @"vectorSize": [NSNumber numberWithUnsignedInteger:vector.size]}];
	
	// Creation of sum vector
	MLReal *sumVector= MLAllocRealBuffer(_size);
	
	// Sum of vectors
	ML_VADD(_vector, 1, vector.vector, 1, sumVector, 1, _size);
	
	// Creation of vector object
	return [[MLWordVector alloc] initWithVector:sumVector size:_size freeVectorOnDealloc:YES];
}

- (MLWordVector *) subtractVector:(MLWordVector *)vector {
	
	// Checks
	if (_size != vector.size)
		@throw [MLWordVectorException wordVectorExceptionWithReason:@"Vectors must have the same size"
														   userInfo:@{@"size": [NSNumber numberWithUnsignedInteger:_size],
																	  @"vectorSize": [NSNumber numberWithUnsignedInteger:vector.size]}];
	
	// Creation of difference vector
	MLReal *subVector= MLAllocRealBuffer(_size);
	
	// Subtraction of vectors
	ML_VSUB(vector.vector, 1, _vector, 1, subVector, 1, _size);
	
	// Creation of vector object
	return [[MLWordVector alloc] initWithVector:subVector size:_size freeVectorOnDealloc:YES];
}

- (MLReal) similarityToVector:(MLWordVector *)vector {
	
	// Checks
	if (_size != vector.size)
		@throw [MLWordVectorException wordVectorExceptionWithReason:@"Vectors must have the same size"
														   userInfo:@{@"size": [NSNumber numberWithUnsignedInteger:_size],
																	  @"vectorSize": [NSNumber numberWithUnsignedInteger:vector.size]}];
	
	// If one of magnitues is 0 return 0 (i.e. orthogonality)
	if ((_magnitude * vector.magnitude) == 0.0)
		return 0.0;
	
	// Dot product of vectors
	MLReal dot= 0.0;
	ML_DOTPR(_vector, 1, vector.vector, 1, &dot, _size);
	
	// Return cosine similarity
	return dot / (_magnitude * vector.magnitude);
}

- (MLReal) distanceToVector:(MLWordVector *)vector {
	
	// Checks
	if (_size != vector.size)
		@throw [MLWordVectorException wordVectorExceptionWithReason:@"Vectors must have the same size"
														   userInfo:@{@"size": [NSNumber numberWithUnsignedInteger:_size],
																	  @"vectorSize": [NSNumber numberWithUnsignedInteger:vector.size]}];
	
	// Allocate temp vector
    MLReal *temp= MLAllocRealBuffer(_size);

	// Subtraction of vectors
	ML_VSUB(vector.vector, 1, _vector, 1, temp, 1, _size);
	
	// Compute magnitude of vector difference
	MLReal distance= 0.0;
	ML_SVESQ(temp, 1, &distance, _size);
	distance= ML_SQRT(distance);
    
    MLFreeRealBuffer(temp);
	
	return distance;
}


#pragma -
#pragma Object overrides

- (BOOL) isEqual:(id)object {
    if (![object isKindOfClass:[MLWordVector class]])
        @throw [MLWordVectorException wordVectorExceptionWithReason:@"Trying to compare an MLWordVector with something else"
                                                           userInfo:@{@"self": self,
                                                                      @"object": object}];
    
    MLWordVector *otherVector= (MLWordVector *) object;
    
    // First check size
    if (_size != otherVector.size)
        return NO;
    
    // Then check magnitude
    if (_magnitude != otherVector.magnitude)
        return NO;
    
    // Allocate temp vector, if needed
    MLReal *temp= MLAllocRealBuffer(_size);
    
    // Finally check the numbers, we subtract the vectors
    // and sum the results
    ML_VSUB(otherVector.vector, 1, _vector, 1, temp, 1, _size);

    MLReal sum= 0.0;
    ML_SVE(temp, 1, &sum, _size);
    
    MLFreeRealBuffer(temp);

    return (sum == 0.0);
}

- (NSUInteger) hash {
    
    // We sum the vector and form an hash box XOR-ing
    // the integer and fractional part projected to UINT_MAX
    MLReal sum= 0.0;
    ML_SVE(_vector, 1, &sum, _size);
    
    double integer= 0.0;
    double fraction= modf(sum, &integer);
    
    NSUInteger hash= (NSUInteger) (fraction * ((double) UINT_MAX));
    hash= hash ^ (NSUInteger) remainder(integer, (double) UINT_MAX);

    return hash;
}


#pragma -
#pragma Properties

@synthesize vector= _vector;
@synthesize size= _size;
@synthesize magnitude= _magnitude;


@end

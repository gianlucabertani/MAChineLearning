//
//  MLRandom.m
//  MAChineLearning
//
//  Created by Gianluca Bertani on 26/05/15.
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

#import "MLRandom.h"

#import "MLConstants.h"

#import <Accelerate/Accelerate.h>

#define RANDOMIZATION_EXCEPTION_NAME          (@"MLRandomException")


#pragma mark -
#pragma mark Static constants

static const MLReal __pi=                   M_PI;
static const MLReal __two=                  2.0;
static const MLReal __one=                  1.0;
static const MLReal __minusTwo=            -2.0;


#pragma mark -
#pragma mark Static variables

static MLReal __spareGaussian=        0.0;


#pragma mark -
#pragma mark MLRandom implementation

@implementation MLRandom


+ (NSUInteger) nextUniformUInt {
	NSUInteger random= 0;
	
	int result= SecRandomCopyBytes(kSecRandomDefault, sizeof(random), (uint8_t *) &random);
    if (result != 0)
        @throw [NSException exceptionWithName:RANDOMIZATION_EXCEPTION_NAME
                                       reason:@"Non zero result from SecRandomCopyBytes"
                                     userInfo:@{@"result": [NSNumber numberWithInt:result]}];
	
	return random;
}

+ (NSUInteger) nextUniformUIntWithMax:(NSUInteger)max {
	NSUInteger random= [MLRandom nextUniformUInt];

	return (random % max);
}

+ (MLReal) nextUniformReal {
	
	// Use 23 bits to take maximum advatange of the float's mantissa,
	// in case MLReal is double we waste some bits of its 52 bit mantissa
	MLReal random= (MLReal) [MLRandom nextUniformUIntWithMax:(1 << 23)];
	return (random * (1.0 / ((MLReal) ((1 << 23) - 1))));
}

+ (MLReal) nextUniformRealWithMin:(MLReal)min max:(MLReal)max {
	MLReal random= [MLRandom nextUniformReal];
	return ((random * (max - min)) + min);
}

+ (void) fillVector:(MLReal *)vector size:(NSUInteger)size ofUniformRealsWithMin:(MLReal)min max:(MLReal)max {
	
	// Allocate and fill a temp integer vector
	int *tempUniform= NULL;
	int err= posix_memalign((void **) &tempUniform,
							BUFFER_MEMORY_ALIGNMENT,
							sizeof(int) * size);
	if (err)
		@throw [NSException exceptionWithName:RANDOMIZATION_EXCEPTION_NAME
									   reason:@"Error while allocating buffer"
									 userInfo:@{@"buffer": @"tempUniform",
												@"error": [NSNumber numberWithInt:err]}];
	
	int result= SecRandomCopyBytes(kSecRandomDefault, sizeof(int) * size, (uint8_t *) tempUniform);
    if (result != 0)
        @throw [NSException exceptionWithName:RANDOMIZATION_EXCEPTION_NAME
                                       reason:@"Non zero result from SecRandomCopyBytes"
                                     userInfo:@{@"result": [NSNumber numberWithInt:result]}];
    
	
	// Use vector absolute value to avoid signed randoms
	vDSP_vabsi(tempUniform, 1, tempUniform, 1, size);
	
	// Get the division factor to reduce significant digits
	// to the same size of float's mantissa (23 bits)
	int intBits= (sizeof(int) * 8) -1;
	int intDivisor= 1 << (intBits - 23);

	// Use vector integer division
	vDSP_vsdivi(tempUniform, 1, &intDivisor, tempUniform, 1, size);
	
	// Convert the vector to floating point
	ML_VDSP_VFLT32(tempUniform, 1, vector, 1, size);
	
	// Scale randoms in range 0..1
	MLReal factor= 1.0 / ((MLReal) ((1 << 23) - 1));
	ML_VDSP_VSMUL(vector, 1, &factor, vector, 1, size);
	
	// Finally apply limits
	MLReal delta= max - min;
	ML_VDSP_VSMUL(vector, 1, &delta, vector, 1, size);
	ML_VDSP_VSADD(vector, 1, &min, vector, 1, size);
	
	// Free the temp vector
	free(tempUniform);
	tempUniform= NULL;
}

+ (MLReal) nextGaussianRealWithMean:(MLReal)mean sigma:(MLReal)sigma {
	MLReal gaussian= 0.0;
	
	if (__spareGaussian == 0.0) {
		
		// Get a pair of random numbers
		MLReal random1= [MLRandom nextUniformReal];
		MLReal random2= [MLRandom nextUniformReal];
		
		// Apple Box–Muller transform
		MLReal r= ML_SQRT(-2.0 * ML_LOG(random1));
		MLReal phi= 2.0 * M_PI * random2;
		
		gaussian= r * ML_COS(phi) * sigma + mean;
		__spareGaussian= r * ML_SIN(phi);
	
	} else {
		gaussian= __spareGaussian * sigma + mean;
		__spareGaussian= 0.0;
	}
	
	return gaussian;
}

+ (MLReal) nextFastGaussianRealWithMean:(MLReal)mean sigma:(MLReal)sigma {
	MLReal gaussian= 0.0;
	
	if (__spareGaussian == 0.0) {
		
		// Get a pair of (fast) random numbers
		MLReal random1= [MLRandom nextUniformReal];
		MLReal random2= [MLRandom nextUniformReal];
		
		// Apple Box–Muller transform
		MLReal r= ML_SQRT(-2.0 * ML_LOG(random1));
		MLReal phi= 2.0 * M_PI * random2;
		
		gaussian= r * ML_COS(phi) * sigma + mean;
		__spareGaussian= r * ML_SIN(phi);
		
	} else {
		gaussian= __spareGaussian * sigma + mean;
		__spareGaussian= 0.0;
	}
	
	return gaussian;
}

+ (void) fillVector:(MLReal *)vector size:(NSUInteger)size ofGaussianRealsWithMean:(MLReal)mean sigma:(MLReal)sigma {
	
	// When using strides the size must be scaled, but we have
	// to consider the last element if the original size is odd:
	// - if we have e.g. 3 elements and want to use a stride of 2,
	//   we need a size of 2 (not 1), or the element at [2] will
	//   not be considered
	// - but if we want to use a stride of 2 on odd elements
	//   (i.e. starting with &vector[1]), we need a size of 1,
	//   or it tries to consider element [3], which does not exist
	NSUInteger evenStridedSize= (size % 2 == 0) ? (size / 2) : ((size / 2) +1);
	NSUInteger oddStridedSize= (size % 2 == 0) ? evenStridedSize : (evenStridedSize -1);
	
	// Allocate and fill a temp vectors with uniform randoms
	MLReal *tempGaussian1= NULL;
	int err= posix_memalign((void **) &tempGaussian1,
							BUFFER_MEMORY_ALIGNMENT,
							sizeof(MLReal) * size);
	if (err)
		@throw [NSException exceptionWithName:RANDOMIZATION_EXCEPTION_NAME
									   reason:@"Error while allocating buffer"
									 userInfo:@{@"buffer": @"tempGaussian1",
												@"error": [NSNumber numberWithInt:err]}];

	[MLRandom fillVector:tempGaussian1 size:size ofUniformRealsWithMin:0.0 max:1.0];
	
	if (size > 1) {
		
		// Copy even elements on odd elements, so that
		// we have pairs of identical random numbers
		ML_VDSP_VSMUL(tempGaussian1, 2, &__one, &tempGaussian1[1], 2, oddStridedSize);
	}
	
	MLReal *tempGaussian2= NULL;
	err= posix_memalign((void **) &tempGaussian2,
						BUFFER_MEMORY_ALIGNMENT,
						sizeof(MLReal) * size);
	if (err)
		@throw [NSException exceptionWithName:RANDOMIZATION_EXCEPTION_NAME
									   reason:@"Error while allocating buffer"
									 userInfo:@{@"buffer": @"tempGaussian2",
												@"error": [NSNumber numberWithInt:err]}];
	
	// Duplicate temp1 on temp2
	ML_VDSP_VSMUL(tempGaussian1, 1, &__one, tempGaussian2, 1, size);
	
	// Now fill vector with uniform random numbers, and copy
	// even elements on odd elements as we have done with temp1
	[MLRandom fillVector:vector size:size ofUniformRealsWithMin:0.0 max:1.0];
	
	if (size > 1)
		ML_VDSP_VSMUL(vector, 2, &__one, &vector[1], 2, oddStridedSize);
	
	// Now we jave two sets of random numbers, with one set duplicated on
	// two separate temp vectors, and both sets have numbers duplicated
	// on even and odd elements: we can proceed with Box-Muller transform
	
	// An "int" size is needed by vvlog and vvsqrt
	int intSize= (int) size;

	// Compute first term of Box-Muller transform
	// directly on the vector
	ML_VVLOG(vector, vector, &intSize);
	ML_VDSP_VSMUL(vector, 1, &__minusTwo, vector, 1, size);
	ML_VVSQRT(vector, vector, &intSize);
	
	// Compute the second term of Box-Muller transform
	// on temp vectors, cosine for temp1 and sine for temp2
	ML_VDSP_VSMUL(tempGaussian1, 1, &__pi, tempGaussian1, 1, size);
	ML_VDSP_VSMUL(tempGaussian1, 1, &__two, tempGaussian1, 1, size);
	ML_VVCOS(tempGaussian1, tempGaussian1, &intSize);
	
	ML_VDSP_VSMUL(tempGaussian2, 1, &__pi, tempGaussian2, 1, size);
	ML_VDSP_VSMUL(tempGaussian2, 1, &__two, tempGaussian2, 1, size);
	ML_VVSIN(tempGaussian2, tempGaussian2, &intSize);
	
	// Now form gaussian random numbers by multipling vector by temp1
	// on even indexes and vector by temp2 on odd indexes
	ML_VDSP_VMUL(vector, 2, tempGaussian1, 2, vector, 2, evenStridedSize);
	ML_VDSP_VMUL(&vector[1], 2, tempGaussian2, 2, &vector[1], 2, oddStridedSize);
	
	// Finally apply limits
	ML_VDSP_VSMUL(vector, 1, &sigma, vector, 1, size);
	ML_VDSP_VSADD(vector, 1, &mean, vector, 1, size);
	
	// Free the temp vectors
	free(tempGaussian1);
	tempGaussian1= NULL;
	
	free(tempGaussian2);
	tempGaussian2= NULL;
}


@end

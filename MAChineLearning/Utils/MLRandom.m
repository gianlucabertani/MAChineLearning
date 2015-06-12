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

#import <Accelerate/Accelerate.h>


#pragma mark -
#pragma mark Statics

static MLReal __bifurcationFactor=    3.995;
static MLReal __one=                  1.0;
static MLReal __minusOne=            -1.0;
static MLReal __minusTwo=            -2.0;

static MLReal __lastFastUniformReal=  0.0;
static MLReal __spareGaussian=        0.0;


#pragma mark -
#pragma mark MLRandom implementation

@implementation MLRandom


+ (NSUInteger) nextUniformUInt {
	NSUInteger random= 0;
	
	SecRandomCopyBytes(kSecRandomDefault, sizeof(random), (uint8_t *) &random);
	
	return random;
}

+ (NSUInteger) nextUniformUIntWithMax:(NSUInteger)max {
	NSUInteger random= [MLRandom nextUniformUInt];

	return (random % max);
}

+ (MLReal) nextUniformReal {
	if (sizeof(MLReal) == sizeof(float)) {
		
		// Use 23 bits to take maximum advtange of the float's mantissa
		MLReal random= (double) [MLRandom nextUniformUIntWithMax:(1 << 23)];
		return (random * (1.0 / ((MLReal) ((1 << 23) - 1))));
		
	} else if (sizeof(MLReal) == sizeof(double)) {
		
		// Use 52 bits to take maximum advtange of the double's mantissa
		MLReal random= (double) [MLRandom nextUniformUIntWithMax:(1L << 52L)];
		return (random * (1.0 / ((MLReal) ((1L << 52) - 1L))));
	
	} else
		@throw [NSException exceptionWithName:@"MLRandomException"
									   reason:@"Unknown size of MLReal type"
									 userInfo:@{@"sizeOfMLReal": [NSNumber numberWithInt:sizeof(MLReal)]}];
}

+ (MLReal) nextUniformRealWithMin:(MLReal)min max:(MLReal)max {
	MLReal random= [MLRandom nextUniformReal];
	
	return ((random * (max - min)) + min);
}

+ (MLReal) nextFastUniformReal {
	if (__lastFastUniformReal == 0.0) {
		
		// Use Secure Random for the seed
		__lastFastUniformReal= [MLRandom nextUniformReal];
		
	} else {
		
		// Use the bifurcation function for subsequent iteratios
		MLReal dummy= 0.0;
		__lastFastUniformReal= ML_MODF(__bifurcationFactor * (1.0 - __lastFastUniformReal), &dummy);
	}
	
	return __lastFastUniformReal;
}

+ (void) fillVector:(MLReal *)vector size:(NSUInteger)size ofUniformRealWithMin:(MLReal)min max:(MLReal)max {
	
	// First fill the vector with a ramp between a random min
	// and max (may be inverted, it's fine)
	if (size > 1) {
		MLReal randomMin= [MLRandom nextFastUniformReal];
		MLReal randomMax= [MLRandom nextFastUniformReal];
		ML_VDSP_VGEN(&randomMin, &randomMax, vector, 1, size);

	} else
		vector[0]= [MLRandom nextFastUniformReal];
	
	// Apply the bifurcation function
	ML_VDSP_VSMUL(vector, 1, &__minusOne, vector, 1, size);
	ML_VDSP_VSADD(vector, 1, &__one, vector, 1, size);
	ML_VDSP_VSMUL(vector, 1, &__bifurcationFactor, vector, 1, size);
	ML_VDSP_VFRAC(vector, 1, vector, 1, size);
	
	// Finally apply limits
	MLReal delta= max - min;
	ML_VDSP_VSMUL(vector, 1, &delta, vector, 1, size);
	ML_VDSP_VSADD(vector, 1, &min, vector, 1, size);
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
		MLReal random1= [MLRandom nextFastUniformReal];
		MLReal random2= [MLRandom nextFastUniformReal];
		
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

+ (void) fillVector:(MLReal *)vector size:(NSUInteger)size ofGaussianRealWithMean:(MLReal)mean sigma:(MLReal)sigma {
	
	// First fill the vector with uniform randoms
	[MLRandom fillVector:vector size:size ofUniformRealWithMin:0.0 max:1.0];
	
	// An "int" size is needed by vvlog and vvsqrt
	int intSize= (int) size;

	// Compute first term of Box-Muller transform
	// directly in the vector
	ML_VVLOG(vector, vector, &intSize);
	ML_VDSP_VSMUL(vector, 1, &__minusTwo, vector, 1, size);
	ML_VVSQRT(vector, vector, &intSize);
	
	// Get a new uniform random number to be used
	// as a second term of Box–Muller transform
	MLReal random= [MLRandom nextFastUniformReal];
	MLReal cosPhi= ML_COS(2.0 * M_PI * random);
	ML_VDSP_VSMUL(vector, 1, &cosPhi, vector, 1, size);
	
	// Finally apply limits
	ML_VDSP_VSMUL(vector, 1, &sigma, vector, 1, size);
	ML_VDSP_VSADD(vector, 1, &mean, vector, 1, size);
}


@end

//
//  RandomTests.m
//  MAChineLearning
//
//  Created by Gianluca Bertani on 12/06/15.
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

#import <Cocoa/Cocoa.h>
#import <XCTest/XCTest.h>
#import <MAChineLearning/MAChineLearning.h>

#define UNIFORM_TEST_SIZE              (10000)
#define GAUSSIAN_TEST_SIZE             (10000)


#pragma mark -
#pragma mark RandomTests declaration

@interface RandomTests : XCTestCase
@end


#pragma mark -
#pragma mark RandomTests implementation

@implementation RandomTests


#pragma mark -
#pragma mark Setup and tear down

- (void) setUp {
    [super setUp];
}

- (void) tearDown {
    [super tearDown];
}


#pragma mark -
#pragma mark Tests

- (void) testUniform {
	@try {
		MLReal *distribution= NULL;
		
		int err= posix_memalign((void **) &distribution,
								BUFFER_MEMORY_ALIGNMENT,
								sizeof(MLReal) * UNIFORM_TEST_SIZE);
		if (err)
			@throw [NSException exceptionWithName:@"PosixMemalignError"
										   reason:@"Error while allocating buffer"
										 userInfo:@{@"buffer": @"distribution",
													@"error": [NSNumber numberWithInt:err]}];
		
		MLReal randomMin= [MLRandom nextUniformReal] / 2.0;
		MLReal randomMax= 0.5 + [MLRandom nextUniformReal] / 2.0;
		[MLRandom fillVector:distribution size:UNIFORM_TEST_SIZE ofUniformRealWithMin:randomMin max:randomMax];
		
		MLReal min= 1.0, max= 0.0, sum= 0.0;;
		for (int i= 0; i < UNIFORM_TEST_SIZE; i++) {
			if (distribution[i] < min)
				min= distribution[i];
			
			if (distribution[i] > max)
				max= distribution[i];
			
			sum += distribution[i];
		}
		
		MLReal mean= sum / ((MLReal) UNIFORM_TEST_SIZE);
		
		MLReal variance= 0.0;
		for (int i= 0; i < UNIFORM_TEST_SIZE; i++)
			variance += (distribution[i] - mean) * (distribution[i] - mean);
		
		variance /= ((MLReal) UNIFORM_TEST_SIZE);
		MLReal sigma= ML_SQRT(variance);
		
		XCTAssertEqualWithAccuracy(min, randomMin, 0.02);
		XCTAssertEqualWithAccuracy(max, randomMax, 0.02);
		XCTAssertEqualWithAccuracy(mean, (randomMin + randomMax) / 2.0, 0.02);
		XCTAssertEqualWithAccuracy(sigma, (randomMax - randomMin) / ML_SQRT(12.0), 0.02);
	
	} @catch (NSException *e) {
		XCTFail(@"Exception caught while testing: %@, reason: '%@', user info: %@", e.name, e.reason, e.userInfo);
	}
}

- (void) testGaussian {
	@try {
		MLReal *distribution= NULL;
		
		int err= posix_memalign((void **) &distribution,
								BUFFER_MEMORY_ALIGNMENT,
								sizeof(MLReal) * GAUSSIAN_TEST_SIZE);
		if (err)
			@throw [NSException exceptionWithName:@"PosixMemalignError"
										   reason:@"Error while allocating buffer"
										 userInfo:@{@"buffer": @"distribution",
													@"error": [NSNumber numberWithInt:err]}];
		
		MLReal randomMean= [MLRandom nextUniformReal] - 0.5;
		MLReal randomSigma= [MLRandom nextUniformReal];
		[MLRandom fillVector:distribution size:GAUSSIAN_TEST_SIZE ofGaussianRealWithMean:randomMean sigma:randomSigma];
		
		MLReal sum= 0.0;;
		for (int i= 0; i < GAUSSIAN_TEST_SIZE; i++)
			sum += distribution[i];
		
		MLReal mean= sum / ((MLReal) GAUSSIAN_TEST_SIZE);
		
		MLReal variance= 0.0;
		for (int i= 0; i < GAUSSIAN_TEST_SIZE; i++)
			variance += (distribution[i] - mean) * (distribution[i] - mean);
		
		variance /= ((MLReal) GAUSSIAN_TEST_SIZE);
		MLReal sigma= ML_SQRT(variance);
		
		XCTAssertEqualWithAccuracy(mean, randomMean, 0.01);
		XCTAssertEqualWithAccuracy(sigma, randomSigma, 0.01);
		
	} @catch (NSException *e) {
		XCTFail(@"Exception caught while testing: %@, reason: '%@', user info: %@", e.name, e.reason, e.userInfo);
	}
}


@end

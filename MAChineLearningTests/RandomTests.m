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

#define UNIFORM_TEST_SIZE              (1000)
#define GAUSSIAN_TEST_SIZE             (1000)

#define DUMP_INTERVALS                  (100)


#pragma mark -
#pragma mark RandomTests declaration

@interface RandomTests : XCTestCase


#pragma mark -
#pragma mark Utility methods

+ (void) dumpDistribution:(MLReal *)distribution size:(NSUInteger)size;


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
		MLReal *distribution= mlAllocRealBuffer(UNIFORM_TEST_SIZE);
		
		MLReal randomMin= [MLRandom nextUniformReal] * 0.5;
		MLReal randomMax= 0.5 + [MLRandom nextUniformReal] * 0.5;
		[MLRandom fillVector:distribution size:UNIFORM_TEST_SIZE ofUniformRealsWithMin:randomMin max:randomMax];
		
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
		
		XCTAssertEqualWithAccuracy(min, randomMin, 0.05);
		XCTAssertEqualWithAccuracy(max, randomMax, 0.05);
		XCTAssertEqualWithAccuracy(mean, (randomMin + randomMax) / 2.0, 0.05);
		XCTAssertEqualWithAccuracy(sigma, (randomMax - randomMin) / ML_SQRT(12.0), 0.05);
		
		// Uncomment to dump the distribution
//		[RandomTests dumpDistribution:distribution size:UNIFORM_TEST_SIZE];
	
	} @catch (NSException *e) {
		XCTFail(@"Exception caught while testing: %@, reason: '%@', user info: %@\nStack trace:%@", e.name, e.reason, e.userInfo, e.callStackSymbols);
	}
}

- (void) testGaussian {
	@try {
		MLReal *distribution= mlAllocRealBuffer(GAUSSIAN_TEST_SIZE);
		
		MLReal randomMean= [MLRandom nextUniformReal] - 0.5;
		MLReal randomSigma= [MLRandom nextUniformReal];
		[MLRandom fillVector:distribution size:GAUSSIAN_TEST_SIZE ofGaussianRealsWithMean:randomMean sigma:randomSigma];
		
		MLReal min= 1.0, max= 0.0, sum= 0.0;;
		for (int i= 0; i < GAUSSIAN_TEST_SIZE; i++) {
			if (distribution[i] < min)
				min= distribution[i];
			
			if (distribution[i] > max)
				max= distribution[i];

			sum += distribution[i];
		}
		
		MLReal mean= sum / ((MLReal) GAUSSIAN_TEST_SIZE);
		
		MLReal variance= 0.0;
		for (int i= 0; i < GAUSSIAN_TEST_SIZE; i++)
			variance += (distribution[i] - mean) * (distribution[i] - mean);
		
		variance /= ((MLReal) GAUSSIAN_TEST_SIZE);
		MLReal sigma= ML_SQRT(variance);
		
		XCTAssertLessThan(min, randomMean - randomSigma);
		XCTAssertGreaterThan(max, randomMean + randomSigma);
		XCTAssertEqualWithAccuracy(mean, randomMean, 0.05);
		XCTAssertEqualWithAccuracy(sigma, randomSigma, 0.05);
		
		// Uncomment to dump the distribution
//		[RandomTests dumpDistribution:distribution size:GAUSSIAN_TEST_SIZE];
		
	} @catch (NSException *e) {
		XCTFail(@"Exception caught while testing: %@, reason: '%@', user info: %@\nStack trace:%@", e.name, e.reason, e.userInfo, e.callStackSymbols);
	}
}


#pragma mark -
#pragma mark Utility methods

+ (void) dumpDistribution:(MLReal *)distribution size:(NSUInteger)size {
	MLReal min= 1.0, max= 0.0;
	for (int i= 0; i < size; i++) {
		if (distribution[i] < min)
			min= distribution[i];
		
		if (distribution[i] > max)
			max= distribution[i];
	}
	
	int occurrencies[DUMP_INTERVALS], maxOccurrencies= 0;
	for (int j= 0; j < DUMP_INTERVALS; j++)
		occurrencies[j]= 0;
	
	for (int i= 0; i < UNIFORM_TEST_SIZE; i++) {
		int j= DUMP_INTERVALS * (distribution[i] - min) / (max - min);
		if ((j < 0) || (j >= DUMP_INTERVALS))
			continue;
		
		occurrencies[j]++;
		
		if (occurrencies[j] > maxOccurrencies)
			maxOccurrencies= occurrencies[j];
	}
	
	NSMutableString *line= [[NSMutableString alloc] init];
	for (int i= maxOccurrencies; i >= 0; i--) {
		[line setString:@""];
		
		for (int j= 0; j < DUMP_INTERVALS; j++)
			[line appendString:(occurrencies[j] >= i) ? @"#" : @" "];
		
		NSLog(@"%@", line);
	}
}


@end

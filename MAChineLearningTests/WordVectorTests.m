//
//  WordVectorTests.m
//  MAChineLearning
//
//  Created by Gianluca Bertani on 05/06/15.
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


#pragma mark -
#pragma mark WordVectorTests declaration

@interface WordVectorTests : XCTestCase

@end


#pragma mark -
#pragma mark WordVectorTests declaration

@implementation WordVectorTests


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

- (void) testAdd {
	@try {
		MLReal vec1[]= { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		MLReal vec2[]= { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		
		for (int i= 0; i < 10; i++) {
			vec1[i]= [MLRandom nextDouble];
			vec2[i]= [MLRandom nextDouble];
		}
		
		MLWordVector *vector1= [[MLWordVector alloc] initWithVector:vec1 size:10 freeVectorOnDealloc:NO];
		MLWordVector *vector2= [[MLWordVector alloc] initWithVector:vec2 size:10 freeVectorOnDealloc:NO];
		
		MLWordVector *sum= [vector1 addVector:vector2];
		
		for (int i= 0; i < 10; i++)
			XCTAssertEqualWithAccuracy(sum.vector[i], vec1[i] + vec2[i], 0.0000000001);
		
	} @catch (NSException *e) {
		XCTFail(@"Exception caught while testing: %@, reason: '%@', user info: %@", e.name, e.reason, e.userInfo);
	}
}

- (void) testSub {
	@try {
		MLReal vec1[]= { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		MLReal vec2[]= { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		
		for (int i= 0; i < 10; i++) {
			vec1[i]= [MLRandom nextDouble];
			vec2[i]= [MLRandom nextDouble];
		}
		
		MLWordVector *vector1= [[MLWordVector alloc] initWithVector:vec1 size:10 freeVectorOnDealloc:NO];
		MLWordVector *vector2= [[MLWordVector alloc] initWithVector:vec2 size:10 freeVectorOnDealloc:NO];
		
		MLWordVector *sub= [vector1 subtractVector:vector2];
		
		for (int i= 0; i < 10; i++)
			XCTAssertEqualWithAccuracy(sub.vector[i], vec1[i] - vec2[i], 0.0000000001);
		
	} @catch (NSException *e) {
		XCTFail(@"Exception caught while testing: %@, reason: '%@', user info: %@", e.name, e.reason, e.userInfo);
	}
}

- (void) testSimilarity {
	@try {
		MLReal refVec[]= { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		MLReal invVec[]= { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		MLReal ortVec[]= { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		
		for (int i= 0; i < 10; i++) {
			if (i % 2 == 0) {
				refVec[i]= [MLRandom nextDouble];
				invVec[i]= -refVec[i];
			
			} else
				ortVec[i]= [MLRandom nextDouble];
		}
		
		MLWordVector *refVector= [[MLWordVector alloc] initWithVector:refVec size:10 freeVectorOnDealloc:NO];
		MLWordVector *invVector= [[MLWordVector alloc] initWithVector:invVec size:10 freeVectorOnDealloc:NO];
		MLWordVector *ortVector= [[MLWordVector alloc] initWithVector:ortVec size:10 freeVectorOnDealloc:NO];
		
		MLReal opposite1= [refVector similarityToVector:invVector];
		XCTAssertEqualWithAccuracy(opposite1, -1.0, 0.000001);
		
		MLReal opposite2= [invVector similarityToVector:refVector];
		XCTAssertEqualWithAccuracy(opposite2, -1.0, 0.000001);
		
		MLReal equal1= [refVector similarityToVector:refVector];
		XCTAssertEqualWithAccuracy(equal1, 1.0, 0.000001);
		
		MLReal equal2= [invVector similarityToVector:invVector];
		XCTAssertEqualWithAccuracy(equal2, 1.0, 0.000001);
		
		MLReal null1= [refVector similarityToVector:ortVector];
		XCTAssertEqualWithAccuracy(null1, 0.0, 0.000001);
		
		MLReal null2= [ortVector similarityToVector:refVector];
		XCTAssertEqualWithAccuracy(null2, 0.0, 0.000001);
		
		MLReal null3= [invVector similarityToVector:ortVector];
		XCTAssertEqualWithAccuracy(null3, 0.0, 0.000001);
		
		MLReal null4= [ortVector similarityToVector:invVector];
		XCTAssertEqualWithAccuracy(null4, 0.0, 0.000001);
		
	} @catch (NSException *e) {
		XCTFail(@"Exception caught while testing: %@, reason: '%@', user info: %@", e.name, e.reason, e.userInfo);
	}
}


@end

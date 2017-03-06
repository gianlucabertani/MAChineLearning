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


#pragma mark -
#pragma mark Internal

- (void) checkEquivalenceOf:(NSString *)word1 to:(NSString *)word2 with:(NSString *)word3 to:(NSString *)word4 on:(MLWordVectorDictionary *)map;


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
			vec1[i]= [MLRandom nextUniformReal];
			vec2[i]= [MLRandom nextUniformReal];
		}
		
		MLWordVector *vector1= [[MLWordVector alloc] initWithVector:vec1 size:10 freeVectorOnDealloc:NO];
		MLWordVector *vector2= [[MLWordVector alloc] initWithVector:vec2 size:10 freeVectorOnDealloc:NO];
		
		MLWordVector *sum= [vector1 addVector:vector2];
		
		for (int i= 0; i < 10; i++)
			XCTAssertEqualWithAccuracy(sum.vector[i], vec1[i] + vec2[i], 0.0000000001);
		
	} @catch (NSException *e) {
		XCTFail(@"Exception caught while testing: %@, reason: '%@', user info: %@\nStack trace:%@", e.name, e.reason, e.userInfo, e.callStackSymbols);
	}
}

- (void) testSub {
	@try {
		MLReal vec1[]= { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		MLReal vec2[]= { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		
		for (int i= 0; i < 10; i++) {
			vec1[i]= [MLRandom nextUniformReal];
			vec2[i]= [MLRandom nextUniformReal];
		}
		
		MLWordVector *vector1= [[MLWordVector alloc] initWithVector:vec1 size:10 freeVectorOnDealloc:NO];
		MLWordVector *vector2= [[MLWordVector alloc] initWithVector:vec2 size:10 freeVectorOnDealloc:NO];
		
		MLWordVector *sub= [vector1 subtractVector:vector2];
		
		for (int i= 0; i < 10; i++)
			XCTAssertEqualWithAccuracy(sub.vector[i], vec1[i] - vec2[i], 0.0000000001);
		
	} @catch (NSException *e) {
		XCTFail(@"Exception caught while testing: %@, reason: '%@', user info: %@\nStack trace:%@", e.name, e.reason, e.userInfo, e.callStackSymbols);
	}
}

- (void) testSimilarity {
	@try {
		MLReal refVec[]= { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		MLReal invVec[]= { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		MLReal ortVec[]= { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		
		for (int i= 0; i < 10; i++) {
			if (i % 2 == 0) {
				refVec[i]= [MLRandom nextUniformReal];
				invVec[i]= -refVec[i];
			
			} else
				ortVec[i]= [MLRandom nextUniformReal];
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
		XCTFail(@"Exception caught while testing: %@, reason: '%@', user info: %@\nStack trace:%@", e.name, e.reason, e.userInfo, e.callStackSymbols);
	}
}

- (void) testWord2vec {
    @try {
        NSBundle *bundle= [NSBundle bundleForClass:[self class]];
        NSString *word2vecSamplePath= [bundle pathForResource:@"Word2vec-sample" ofType:@"bin"];
        XCTAssertNotNil(word2vecSamplePath);
        
        // Here we use a pre-trained sample vector file, obtained by training with the 6000
        // most used words from the same sample text used by the word2vec demo, we then test
        // the vector map with known working equivalences
        MLWordVectorDictionary *map= [MLWordVectorDictionary createFromWord2vecFile:word2vecSamplePath binary:YES];
        XCTAssertNotNil(map);
        
        // Capital to nation
        [self checkEquivalenceOf:@"philadelphia" to:@"pennsylvania" with:@"miami" to:@"florida" on:map];
        
        // Masculine to feminine
        [self checkEquivalenceOf:@"he" to:@"she" with:@"brother" to:@"sister" on:map];
        [self checkEquivalenceOf:@"son" to:@"daughter" with:@"his" to:@"her" on:map];
        
        // Adjective to adverb
        [self checkEquivalenceOf:@"obvious" to:@"obviously" with:@"sudden" to:@"suddenly" on:map];
        [self checkEquivalenceOf:@"usual" to:@"usually" with:@"typical" to:@"typically" on:map];
        
        // Adjective to superlative
        [self checkEquivalenceOf:@"long" to:@"longest" with:@"low" to:@"lowest" on:map];
        
        // Infinitive to gerund
        [self checkEquivalenceOf:@"code" to:@"coding" with:@"dance" to:@"dancing" on:map];
        [self checkEquivalenceOf:@"sing" to:@"singing" with:@"look" to:@"looking" on:map];
        
        // Nation to nationality
        [self checkEquivalenceOf:@"austria" to:@"austrian" with:@"mexico" to:@"mexican" on:map];
        [self checkEquivalenceOf:@"greece" to:@"greek" with:@"bulgaria" to:@"bulgarian" on:map];
        [self checkEquivalenceOf:@"france" to:@"french" with:@"albania" to:@"albanian" on:map];
        [self checkEquivalenceOf:@"italy" to:@"italian" with:@"denmark" to:@"danish" on:map];
        [self checkEquivalenceOf:@"ireland" to:@"irish" with:@"brazil" to:@"brazilian" on:map];
        [self checkEquivalenceOf:@"korea" to:@"korean" with:@"england" to:@"english" on:map];
        [self checkEquivalenceOf:@"norway" to:@"norwegian" with:@"korea" to:@"korean" on:map];
        [self checkEquivalenceOf:@"spain" to:@"spanish" with:@"norway" to:@"norwegian" on:map];
        
        // Gerund to past
        [self checkEquivalenceOf:@"taking" to:@"took" with:@"writing" to:@"wrote" on:map];
        
        // Singular to plurarl
        [self checkEquivalenceOf:@"building" to:@"buildings" with:@"computer" to:@"computers" on:map];
        [self checkEquivalenceOf:@"horse" to:@"horses" with:@"computer" to:@"computers" on:map];
        [self checkEquivalenceOf:@"dog" to:@"dogs" with:@"computer" to:@"computers" on:map];
        [self checkEquivalenceOf:@"mouse" to:@"mice" with:@"dog" to:@"dogs" on:map];
        
        // First person to third person
        [self checkEquivalenceOf:@"find" to:@"finds" with:@"say" to:@"says" on:map];
        [self checkEquivalenceOf:@"speak" to:@"speaks" with:@"play" to:@"plays" on:map];
        
    } @catch (NSException *e) {
        XCTFail(@"Exception caught while testing: %@, reason: '%@', user info: %@\nStack trace:%@", e.name, e.reason, e.userInfo, e.callStackSymbols);
    }
}

- (void) testGloVe {
    @try {
        NSBundle *bundle= [NSBundle bundleForClass:[self class]];
        NSString *gloveSamplePath= [bundle pathForResource:@"GloVe-sample" ofType:@"txt"];
        XCTAssertNotNil(gloveSamplePath);
        
        // Here we use a pre-trained sample vector file, obtained by training with the 6000
        // most used words from the same sample text used by the GloVe demo, we then test
        // the vector map with known working equivalences
        MLWordVectorDictionary *map= [MLWordVectorDictionary createFromGloVeFile:gloveSamplePath];
        XCTAssertNotNil(map);
        
        // Capital to nation
        [self checkEquivalenceOf:@"philadelphia" to:@"pennsylvania" with:@"miami" to:@"florida" on:map];
        
        // Masculine to feminine
        [self checkEquivalenceOf:@"he" to:@"she" with:@"brother" to:@"sister" on:map];
        [self checkEquivalenceOf:@"son" to:@"daughter" with:@"his" to:@"her" on:map];
        
        // Adjective to adverb
        [self checkEquivalenceOf:@"obvious" to:@"obviously" with:@"sudden" to:@"suddenly" on:map];
        [self checkEquivalenceOf:@"usual" to:@"usually" with:@"typical" to:@"typically" on:map];
        
        // Adjective to superlative
        [self checkEquivalenceOf:@"long" to:@"longest" with:@"low" to:@"lowest" on:map];
        
        // Infinitive to gerund
        [self checkEquivalenceOf:@"code" to:@"coding" with:@"dance" to:@"dancing" on:map];
        [self checkEquivalenceOf:@"sing" to:@"singing" with:@"look" to:@"looking" on:map];
        
        // Nation to nationality
        [self checkEquivalenceOf:@"austria" to:@"austrian" with:@"mexico" to:@"mexican" on:map];
        [self checkEquivalenceOf:@"greece" to:@"greek" with:@"bulgaria" to:@"bulgarian" on:map];
        [self checkEquivalenceOf:@"france" to:@"french" with:@"albania" to:@"albanian" on:map];
        [self checkEquivalenceOf:@"italy" to:@"italian" with:@"denmark" to:@"danish" on:map];
        [self checkEquivalenceOf:@"ireland" to:@"irish" with:@"brazil" to:@"brazilian" on:map];
        [self checkEquivalenceOf:@"korea" to:@"korean" with:@"england" to:@"english" on:map];
        [self checkEquivalenceOf:@"norway" to:@"norwegian" with:@"korea" to:@"korean" on:map];
        [self checkEquivalenceOf:@"spain" to:@"spanish" with:@"norway" to:@"norwegian" on:map];
        
        // Gerund to past
        [self checkEquivalenceOf:@"taking" to:@"took" with:@"writing" to:@"wrote" on:map];
        
        // Singular to plurarl
        [self checkEquivalenceOf:@"building" to:@"buildings" with:@"computer" to:@"computers" on:map];
        [self checkEquivalenceOf:@"horse" to:@"horses" with:@"computer" to:@"computers" on:map];
        [self checkEquivalenceOf:@"dog" to:@"dogs" with:@"computer" to:@"computers" on:map];
        [self checkEquivalenceOf:@"mouse" to:@"mice" with:@"dog" to:@"dogs" on:map];
        
        // First person to third person
        [self checkEquivalenceOf:@"find" to:@"finds" with:@"say" to:@"says" on:map];
        [self checkEquivalenceOf:@"speak" to:@"speaks" with:@"play" to:@"plays" on:map];
        
    } @catch (NSException *e) {
        XCTFail(@"Exception caught while testing: %@, reason: '%@', user info: %@\nStack trace:%@", e.name, e.reason, e.userInfo, e.callStackSymbols);
    }
}


#pragma mark -
#pragma mark Internal

- (void) checkEquivalenceOf:(NSString *)word1 to:(NSString *)word2 with:(NSString *)word3 to:(NSString *)word4 on:(MLWordVectorDictionary *)map {
    MLWordVector *vec1= [map vectorForWord:word1];
    XCTAssertNotNil(vec1);

    MLWordVector *vec2= [map vectorForWord:word2];
    XCTAssertNotNil(vec2);

    MLWordVector *vec3= [map vectorForWord:word3];
    XCTAssertNotNil(vec3);
    
    MLWordVector *expectedVec4= [map vectorForWord:word4];
    XCTAssertNotNil(expectedVec4);

    MLWordVector *vec4= [[vec2 subtractVector:vec1] addVector:vec3];
    XCTAssertNotNil(vec4);
    
    // Original accuracy checkers of both word2vec and GloVe do not
    // consider first three words, so here we look if the expected
    // resulting word is within the first four similar words
    NSArray *similarWords= [map mostSimilarWordsToVector:vec4];
    NSUInteger index= [similarWords indexOfObject:word4 inRange:NSMakeRange(0, 4)];
    XCTAssertTrue(index != NSNotFound);
    
    if (index == NSNotFound)
        NSLog(@"Failed equivalence for %@:%@ = %@:%@", word1, word2, word3, word4);
}


@end

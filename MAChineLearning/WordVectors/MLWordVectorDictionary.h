//
//  MLWordVectorDictionary.h
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

#import <Foundation/Foundation.h>

#import "MLReal.h"
#import "MLWordExtractorType.h"
#import "MLWordExtractorOption.h"


@class MLNeuralNetwork;
@class MLWordDictionary;
@class MLWordVector;

@interface MLWordVectorDictionary : NSObject


#pragma mark -
#pragma mark Initialization

+ (MLWordVectorDictionary *) createFromWord2vecFile:(NSString *)vectorFilePath binary:(BOOL)binary;
+ (MLWordVectorDictionary *) createFromGloVeFile:(NSString *)vectorFilePath;
+ (MLWordVectorDictionary *) createFromFastTextFile:(NSString *)vectorFilePath;

- (instancetype) initWithDictionary:(NSDictionary *)vectorDictionary;


#pragma mark -
#pragma mark Word lookup and comparison

- (BOOL) containsWord:(NSString *)word;
- (MLWordVector *) vectorForWord:(NSString *)word;

- (NSString *) mostSimilarWordToVector:(MLWordVector *)vector;
- (NSString *) nearestWordToVector:(MLWordVector *)vector;

- (NSArray *) mostSimilarWordsToVector:(MLWordVector *)vector;
- (NSArray *) nearestWordsToVector:(MLWordVector *)vector;


#pragma mark -
#pragma mark Sentence lookup

- (MLWordVector *) vectorForSentence:(NSString *)sentence;
- (MLWordVector *) vectorForSentence:(NSString *)sentence withLanguage:(NSString *)languageCode;
- (MLWordVector *) vectorForSentence:(NSString *)sentence withLanguage:(NSString *)languageCode extractorType:(MLWordExtractorType)extractorType options:(MLWordExtractorOption)options wordNotFound:(void (^)(NSString *))wordNotFoundHandler;


#pragma mark -
#pragma mark Properties

@property (nonatomic, readonly) NSUInteger wordCount;
@property (nonatomic, readonly) NSUInteger vectorSize;


@end

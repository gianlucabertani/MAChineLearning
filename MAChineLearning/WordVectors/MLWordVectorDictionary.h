//
//  MLWordVectorDictionary.h
//  MAChineLearning
//
//  Created by Gianluca Bertani on 03/06/15.
//  Copyright (c) 2015-2018 Gianluca Bertani. All rights reserved.
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


typedef void (^MLWordNotFoundHanlder)(NSString * _Nonnull word);


@class MLNeuralNetwork;
@class MLWordDictionary;
@class MLWordVector;

@interface MLWordVectorDictionary : NSObject


#pragma mark -
#pragma mark Initialization

+ (nonnull MLWordVectorDictionary *) createFromWord2vecFile:(nonnull NSString *)vectorFilePath binary:(BOOL)binary;
+ (nonnull MLWordVectorDictionary *) createFromGloVeFile:(nonnull NSString *)vectorFilePath;
+ (nonnull MLWordVectorDictionary *) createFromFastTextFile:(nonnull NSString *)vectorFilePath;

- (nonnull instancetype) init NS_UNAVAILABLE;

- (nonnull instancetype) initWithDictionary:(nonnull NSDictionary<NSString *, MLWordVector *> *)vectorDictionary NS_DESIGNATED_INITIALIZER;


#pragma mark -
#pragma mark Word lookup and comparison

- (BOOL) containsWord:(nonnull NSString *)word;
- (nullable MLWordVector *) vectorForWord:(nonnull NSString *)word;

- (nullable NSString *) mostSimilarWordToVector:(nonnull MLWordVector *)vector;
- (nullable NSString *) nearestWordToVector:(nonnull MLWordVector *)vector;

- (nonnull NSArray<NSString *> *) mostSimilarWordsToVector:(nonnull MLWordVector *)vector;
- (nonnull NSArray<NSString *> *) nearestWordsToVector:(nonnull MLWordVector *)vector;


#pragma mark -
#pragma mark Sentence lookup

- (nonnull MLWordVector *) vectorForSentence:(nonnull NSString *)sentence;

- (nonnull MLWordVector *) vectorForSentence:(nonnull NSString *)sentence
                                withLanguage:(nullable NSString *)languageCode;

- (nonnull MLWordVector *) vectorForSentence:(nonnull NSString *)sentence
                                withLanguage:(nullable NSString *)languageCode
                               extractorType:(MLWordExtractorType)extractorType
                                     options:(MLWordExtractorOption)options
                                wordNotFound:(MLWordNotFoundHanlder)wordNotFoundHandler;


#pragma mark -
#pragma mark Properties

@property (nonatomic, readonly) NSUInteger wordCount;
@property (nonatomic, readonly) NSUInteger vectorSize;


@end

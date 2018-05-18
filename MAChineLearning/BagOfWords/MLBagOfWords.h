//
//  MLBagOfWords.h
//  MAChineLearning
//
//  Created by Gianluca Bertani on 23/04/15.
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
#import "MLFeatureNormalizationType.h"


@class MLWordDictionary;
@class MLMutableWordDictionary;

@interface MLBagOfWords : NSObject


#pragma mark -
#pragma mark Initialization

+ (nonnull MLBagOfWords *) bagOfWordsForTopicClassificationWithText:(nonnull NSString *)text
                                                         documentID:(nullable NSString *)documentID
                                                         dictionary:(nonnull MLMutableWordDictionary *)dictionary
                                                           language:(nonnull NSString *)languageCode
                                               featureNormalization:(MLFeatureNormalizationType)normalizationType;

+ (nonnull MLBagOfWords *) bagOfWordsForSentimentAnalysisWithText:(nonnull NSString *)text
                                                       documentID:(nullable NSString *)documentID
                                                       dictionary:(nonnull MLMutableWordDictionary *)dictionary
                                                         language:(nonnull NSString *)languageCode
                                             featureNormalization:(MLFeatureNormalizationType)normalizationType;

+ (nonnull MLBagOfWords *) bagOfWordsWithText:(nonnull NSString *)text
                                   documentID:(nullable NSString *)documentID
                                   dictionary:(nonnull MLWordDictionary *)dictionary
                              buildDictionary:(BOOL)buildDictionary
                                     language:(nonnull NSString *)languageCode
                                wordExtractor:(MLWordExtractorType)extractorType
                             extractorOptions:(MLWordExtractorOption)extractorOptions
                         featureNormalization:(MLFeatureNormalizationType)normalizationType
                                 outputBuffer:(nullable MLReal *)outputBuffer;

+ (nonnull MLBagOfWords *) bagOfWordsWithWords:(nonnull NSArray<NSString *> *)words
                                    documentID:(nullable NSString *)documentID
                                    dictionary:(nonnull MLWordDictionary *)dictionary
                               buildDictionary:(BOOL)buildDictionary
                          featureNormalization:(MLFeatureNormalizationType)normalizationType
                                  outputBuffer:(nullable MLReal *)outputBuffer;

- (nonnull instancetype) initWithText:(nonnull NSString *)text
                           documentID:(nullable NSString *)documentID
                           dictionary:(nonnull MLWordDictionary *)dictionary
                      buildDictionary:(BOOL)buildDictionary
                             language:(nonnull NSString *)languageCode
                        wordExtractor:(MLWordExtractorType)extractorType
                     extractorOptions:(MLWordExtractorOption)extractorOptions
                 featureNormalization:(MLFeatureNormalizationType)normalizationType
                         outputBuffer:(nullable MLReal *)outputBuffer;

- (nonnull instancetype) initWithWords:(nonnull NSArray<NSString *> *)words
                            documentID:(nullable NSString *)documentID
                            dictionary:(nonnull MLWordDictionary *)dictionary
                       buildDictionary:(BOOL)buildDictionary
                  featureNormalization:(MLFeatureNormalizationType)normalizationType
                          outputBuffer:(nullable MLReal *)outputBuffer;


#pragma mark -
#pragma mark Dictionary building

+ (void) buildDictionaryWithText:(nonnull NSString *)text
					  documentID:(nullable NSString *)documentID
					  dictionary:(nonnull MLMutableWordDictionary *)dictionary
						language:(nonnull NSString *)languageCode
				   wordExtractor:(MLWordExtractorType)extractorType
				extractorOptions:(MLWordExtractorOption)extractorOptions;


#pragma mark -
#pragma mark Languages guessing

+ (nullable NSString *) guessLanguageCodeWithLinguisticTaggerForText:(nonnull NSString *)text;
+ (nullable NSString *) guessLanguageCodeWithStopWordsForText:(nonnull NSString *)text;


#pragma mark -
#pragma mark Word extractors

+ (nonnull NSArray<NSString *> *) extractWordsWithSimpleTokenizerFromText:(nonnull NSString *)text
                                                             withLanguage:(nonnull NSString *)languageCode
                                                         extractorOptions:(MLWordExtractorOption)extractorOptions;

+ (nonnull NSArray<NSString *> *) extractWordsWithLinguisticTaggerFromText:(nonnull NSString *)text
                                                              withLanguage:(nonnull NSString *)languageCode
                                                          extractorOptions:(MLWordExtractorOption)extractorOptions;


#pragma mark -
#pragma mark Properties

@property (nonatomic, readonly, nullable) NSString *documentID;
@property (nonatomic, readonly, nonnull) NSArray<NSString *> *words;

@property (nonatomic, readonly) NSUInteger outputSize;
@property (nonatomic, readonly, nonnull) MLReal *outputBuffer;


@end

//
//  BagOfWords.h
//  MAChineLearning
//
//  Created by Gianluca Bertani on 23/04/15.
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

#import <Foundation/Foundation.h>

#import "Real.h"

#import "WordExtractorType.h"
#import "WordExtractorOption.h"
#import "FeatureNormalizationType.h"


@class TokenDictionary;

@interface BagOfWords : NSObject


#pragma mark -
#pragma mark Initialization

+ (BagOfWords *) bagOfWordsForTopicClassificationWithText:(NSString *)text textID:(NSString *)textID dictionary:(TokenDictionary *)dictionary language:(NSString *)languageCode featureNormalization:(FeatureNormalizationType)normalizationType;

+ (BagOfWords *) bagOfWordsForSentimentAnalysisWithText:(NSString *)text textID:(NSString *)textID dictionary:(TokenDictionary *)dictionary language:(NSString *)languageCode featureNormalization:(FeatureNormalizationType)normalizationType;

+ (BagOfWords *) bagOfWordsWithText:(NSString *)text textID:(NSString *)textID dictionary:(TokenDictionary *)dictionary buildDictionary:(BOOL)buildDictionary language:(NSString *)languageCode wordExtractor:(WordExtractorType)extractorType extractorOptions:(WordExtractorOption)extractorOptions featureNormalization:(FeatureNormalizationType)normalizationType outputBuffer:(REAL *)outputBuffer;

- (id) initWithText:(NSString *)text textID:(NSString *)textID dictionary:(TokenDictionary *)dictionary buildDictionary:(BOOL)buildDictionary language:(NSString *)languageCode wordExtractor:(WordExtractorType)extractorType extractorOptions:(WordExtractorOption)extractorOptions featureNormalization:(FeatureNormalizationType)normalizationType outputBuffer:(REAL *)outputBuffer;


#pragma mark -
#pragma mark Dictionary building

+ (void) buildDictionaryWithText:(NSString *)text textID:(NSString *)textID dictionary:(TokenDictionary *)dictionary language:(NSString *)languageCode wordExtractor:(WordExtractorType)extractorType extractorOptions:(WordExtractorOption)extractorOptions;


#pragma mark -
#pragma mark Languages guessing

+ (NSString *) guessLanguageCodeWithLinguisticTagger:(NSString *)text;
+ (NSString *) guessLanguageCodeWithStopWords:(NSString *)text;


#pragma mark -
#pragma mark Properties

@property (nonatomic, readonly) NSString *textID;
@property (nonatomic, readonly) NSString *languageCode;

@property (nonatomic, readonly) WordExtractorType extractorType;
@property (nonatomic, readonly) WordExtractorOption extractorOptions;
@property (nonatomic, readonly) FeatureNormalizationType featureNormalization;

@property (nonatomic, readonly) NSUInteger dictionarySize;
@property (nonatomic, readonly) NSArray *tokens;
@property (nonatomic, readonly) REAL *outputBuffer;


@end

//
//  MLWordDictionary.m
//  MAChineLearning
//
//  Created by Gianluca Bertani on 10/05/15.
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

#import "MLWordDictionary.h"
#import "MLWordInfo.h"
#import "MLBagOfWordsException.h"

#import "MLAlloc.h"


@implementation MLWordDictionary


#pragma mark -
#pragma mark Initialization

- (instancetype) init {
    if ((self = [super init])) {
        
        // Initialization
        _dictionary= [[NSMutableDictionary alloc] init];
        
        _totalWords= 0;
        _totalDocuments= 0;
        
        _documentIDs= [[NSMutableSet alloc] init];
        
        _idfWeights= NULL;
        _idfWeightsDirty= YES;
    }
    
    return self;
}

- (instancetype) initWithDictionary:(MLWordDictionary *)dictionary {
    return [self initWithWordInfos:dictionary.wordInfos];
}

- (instancetype) initWithWordInfos:(NSArray<MLWordInfo *> *)wordInfos {
    if ((self = [self init])) {
        
        // Initialization
        for (MLWordInfo *wordInfo in wordInfos) {
            NSString *word= wordInfo.word.lowercaseString;
            
            MLWordInfo *newWordInfo= [[MLWordInfo alloc] initWithWordInfo:wordInfo newPosition:_dictionary.count];
            _dictionary[word]= newWordInfo;

            _totalWords += wordInfo.totalOccurrencies;
            
            for (NSString *documentID in wordInfo.documentIDs) {
                if (![_documentIDs containsObject:documentID]) {
                    [_documentIDs addObject:documentID];
                    _totalDocuments++;
                }
            }
        }
    }
    
    return self;
}

- (void) dealloc {
    MLFreeRealBuffer(_idfWeights);
    _idfWeights= NULL;
}


#pragma mark -
#pragma mark Dictionary access and manipulation

- (BOOL) containsWord:(NSString *)word {
    NSString *lowercaseWord= word.lowercaseString;
    
    return (_dictionary[lowercaseWord] != nil);
}

- (MLWordInfo *) infoForWord:(NSString *)word {
    NSString *lowercaseWord= word.lowercaseString;
    
    return _dictionary[lowercaseWord];
}


#pragma mark -
#pragma mark Dictionary filtering

- (MLWordDictionary *) keepWordsWithHighestOccurrenciesUpToSize:(NSUInteger)size {
    NSArray<MLWordInfo *> *sortedWordInfos= [_dictionary.allValues sortedArrayUsingComparator:^NSComparisonResult(id obj1, id obj2) {
        MLWordInfo *info1= (MLWordInfo *) obj1;
        MLWordInfo *info2= (MLWordInfo *) obj2;
        
        return (info1.totalOccurrencies < info2.totalOccurrencies) ? NSOrderedDescending :
        ((info1.totalOccurrencies > info2.totalOccurrencies) ? NSOrderedAscending : NSOrderedSame);
    }];
    
    NSMutableArray<MLWordInfo *> *newWordInfos= [[NSMutableArray alloc] initWithCapacity:size];
    for (MLWordInfo *wordInfo in sortedWordInfos) {
        if (newWordInfos.count == size)
            break;
        
        [newWordInfos addObject:wordInfo];
    }
    
    return [[MLWordDictionary alloc] initWithWordInfos:newWordInfos];
}

- (MLWordDictionary *) discardWordsWithOccurrenciesLessThan:(NSUInteger)minOccurrencies {
    return [self filterWordsWith:^MLWordFilterOutcome(MLWordInfo *wordInfo) {
        return (wordInfo.totalOccurrencies < minOccurrencies) ? MLWordFilterOutcomeDiscardWord : MLWordFilterOutcomeKeepWord;
    }];
}

- (MLWordDictionary *) discardWordsWithOccurrenciesGreaterThan:(NSUInteger)maxOccurrencies {
    return [self filterWordsWith:^MLWordFilterOutcome(MLWordInfo *wordInfo) {
        return (wordInfo.totalOccurrencies > maxOccurrencies) ? MLWordFilterOutcomeDiscardWord : MLWordFilterOutcomeKeepWord;
    }];
}

- (MLWordDictionary *) filterWordsWith:(MLWordFilter)filter {
    NSMutableArray<MLWordInfo *> *newWordInfos= [[NSMutableArray alloc] init];

    NSArray<MLWordInfo *> *wordInfos= _dictionary.allValues;
    for (MLWordInfo *wordInfo in wordInfos) {
        MLWordFilterOutcome outcome= filter(wordInfo);

        switch (outcome) {
            case MLWordFilterOutcomeDiscardWord:
                break;
                
            case MLWordFilterOutcomeKeepWord:
                [newWordInfos addObject:wordInfo];
                break;
        }
    }
    
    return [[MLWordDictionary alloc] initWithWordInfos:newWordInfos];
}


#pragma mark -
#pragma mark NSObject overrides

- (NSString *) description {
    NSMutableString *descr= [[NSMutableString alloc] initWithCapacity:100 *_dictionary.count];
    
    [descr appendString:@"{\n"];
    
    NSArray<MLWordInfo *> *wordInfos= _dictionary.allValues;
    for (MLWordInfo *wordInfo in wordInfos)
        [descr appendFormat:@"\t'%@': %lu (%lu)\n", wordInfo.word, (unsigned long) wordInfo.totalOccurrencies, (unsigned long) wordInfo.documentOccurrencies];
    
    [descr appendString:@"}"];
    
    return descr.description;
}


#pragma mark -
#pragma mark Properties

@dynamic size;

- (NSUInteger) size {
    return _dictionary.count;
}

@dynamic wordInfos;

- (NSArray<MLWordInfo *> *) wordInfos {
    return _dictionary.allValues;
}

@synthesize totalWords= _totalWords;
@synthesize totalDocuments= _totalDocuments;

@synthesize documentIDs= _documentIDs;

@dynamic idfWeights;

- (MLReal *) idfWeights {
    if (!_idfWeightsDirty)
        return _idfWeights;
    
    if (!_idfWeights)
        _idfWeights= MLAllocRealBuffer(_dictionary.count);
    
    // Clear the IDF buffer
    ML_VCLR(_idfWeights, 1, _dictionary.count);
    
    // Compute inverse document frequency
    for (MLWordInfo *wordInfo in _dictionary) {
        MLReal weight= log(((MLReal) _totalDocuments) / (1.0 + ((MLReal) wordInfo.documentOccurrencies)));
        _idfWeights[wordInfo.position]= weight;
    }
    
    _idfWeightsDirty= NO;
    
    return _idfWeights;
}


@end


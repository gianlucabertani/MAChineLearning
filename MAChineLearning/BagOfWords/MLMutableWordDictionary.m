//
//  MLMutableWordDictionary.m
//  MAChineLearning
//
//  Created by Gianluca Bertani on 02/06/15.
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

#import "MLMutableWordDictionary.h"
#import "MLMutableWordInfo.h"
#import "MLBagOfWordsException.h"


#pragma mark -
#pragma mark MLMutableWordDictionary extension

@interface MLMutableWordDictionary () {
    NSUInteger _maxSize;
}


@end


#pragma mark -
#pragma mark MLMutableWordDictionary implementation

@implementation MLMutableWordDictionary


#pragma mark -
#pragma mark Initialization

+ (MLMutableWordDictionary *) dictionaryWithMaxSize:(NSUInteger)maxSize {
    return [[MLMutableWordDictionary alloc] initWithMaxSize:maxSize];
}

- (instancetype) initWithMaxSize:(NSUInteger)maxSize {
    if ((self = [super init])) {
        
        // Initialization
        _maxSize= maxSize;
    }
    
    return self;
}


#pragma mark -
#pragma mark Dictionary building

- (void) countOccurrenceForWord:(NSString *)word documentID:(NSString *)documentID {
    NSString *lowercaseWord= word.lowercaseString;
    
    MLMutableWordInfo *wordInfo= (MLMutableWordInfo *) _dictionary[lowercaseWord];
    if (!wordInfo) {
        if (_dictionary.count >= _maxSize)
            return;
        
        wordInfo= [[MLMutableWordInfo alloc] initWithWord:word position:_dictionary.count];
        _dictionary[lowercaseWord]= wordInfo;
    }
    
    [wordInfo countOccurrenceForDocumentID:documentID];
    
    if (documentID && (![_documentIDs containsObject:documentID])) {
        [_documentIDs addObject:documentID];
        _totalDocuments++;
    }
    
    _totalWords++;
    
    _idfWeightsDirty= YES;
}


#pragma mark -
#pragma mark Properties

@synthesize maxSize= _maxSize;


@end

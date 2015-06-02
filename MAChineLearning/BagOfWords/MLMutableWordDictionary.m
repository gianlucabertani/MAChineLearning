//
//  MLMutableWordDictionary.m
//  MAChineLearning
//
//  Created by Gianluca Bertani on 02/06/15.
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

#import "MLMutableWordDictionary.h"
#import "MLWordInfo.h"
#import "MLWordInfo+Mutable.h"


@implementation MLMutableWordDictionary


#pragma mark -
#pragma mark Initialization

+ (MLMutableWordDictionary *) dictionaryWithMaxSize:(NSUInteger)maxSize {
	return [[MLMutableWordDictionary alloc] initWithMaxSize:maxSize];
}

- (instancetype) initWithMaxSize:(NSUInteger)maxSize {
	if ((self = [super init])) {
		
		// Initialization
		_dictionary= [[NSMutableDictionary alloc] initWithCapacity:maxSize];
		_maxSize= maxSize;
	}
	
	return self;
}


#pragma mark -
#pragma mark Dictionary building

- (void) countOccurrenceForWord:(NSString *)word textID:(NSString *)textID {
	NSString *lowercaseWord= [word lowercaseString];
	
	MLWordInfo *wordInfo= [_dictionary objectForKey:lowercaseWord];
	if ((!wordInfo) && (_dictionary.count < _maxSize)) {
		wordInfo= [[MLWordInfo alloc] initWithWord:word position:_dictionary.count];
		[_dictionary setObject:wordInfo forKey:lowercaseWord];
	}
	
	[wordInfo countOccurrenceForTextID:textID];
	
	if (textID && (![_documents containsObject:textID])) {
		[_documents addObject:textID];
		_totalDocuments++;
	}
	
	_totalWords++;
	
	_idfWeightsDirty= YES;
}


#pragma mark -
#pragma mark Dictionary filtering

- (void) keepWordsWithHighestOccurrenciesUpToSize:(NSUInteger)size {
	NSArray *sortedWordInfos= [[_dictionary allValues] sortedArrayUsingComparator:^NSComparisonResult(id obj1, id obj2) {
		MLWordInfo *info1= (MLWordInfo *) obj1;
		MLWordInfo *info2= (MLWordInfo *) obj2;
		
		return (info1.totalOccurrencies < info2.totalOccurrencies) ? NSOrderedDescending :
		((info1.totalOccurrencies > info2.totalOccurrencies) ? NSOrderedAscending : NSOrderedSame);
	}];
	
	NSMutableDictionary *newDictionary= [[NSMutableDictionary alloc] initWithCapacity:_dictionary.count];
	
	for (MLWordInfo *wordInfo in sortedWordInfos) {
		if (newDictionary.count == size)
			break;
		
		NSString *lowercaseWord= [wordInfo.word lowercaseString];
		
		[newDictionary setObject:wordInfo forKey:lowercaseWord];
	}
	
	[_dictionary removeAllObjects];
	_dictionary= newDictionary;
	
	_idfWeightsDirty= YES;
}

- (void) discardWordsWithOccurrenciesLessThan:(NSUInteger)minOccurrencies {
	[self applyFilter:^MLWordFilterOutcome(NSString *word, MLWordInfo *wordInfo) {
		return (wordInfo.totalOccurrencies < minOccurrencies) ? MLWordFilterOutcomeDiscardWord : MLWordFilterOutcomeKeepWord;
	}];
}

- (void) discardWordsWithOccurrenciesGreaterThan:(NSUInteger)maxOccurrencies {
	[self applyFilter:^MLWordFilterOutcome(NSString *word, MLWordInfo *wordInfo) {
		return (wordInfo.totalOccurrencies > maxOccurrencies) ? MLWordFilterOutcomeDiscardWord : MLWordFilterOutcomeKeepWord;
	}];
}

- (void) applyFilter:(MLWordFilter)filter {
	NSArray *words= [_dictionary allKeys];
	
	for (NSString *word in words) {
		MLWordInfo *wordInfo= [_dictionary objectForKey:word];
		
		MLWordFilterOutcome outcome= filter(word, wordInfo);
		switch (outcome) {
			case MLWordFilterOutcomeDiscardWord:
				[_dictionary removeObjectForKey:word];
				break;
				
			case MLWordFilterOutcomeKeepWord:
				break;
		}
	}
	
	_idfWeightsDirty= YES;
}

- (void) compact {
	NSMutableDictionary *newDictionary= [[NSMutableDictionary alloc] initWithCapacity:_dictionary.count];
	
	NSArray *words= [_dictionary allKeys];
	
	for (NSString *word in words) {
		MLWordInfo *wordInfo= [_dictionary objectForKey:word];
		
		MLWordInfo *newWordInfo= [[MLWordInfo alloc] initWithWordInfo:wordInfo newPosition:newDictionary.count];
		[newDictionary setObject:newWordInfo forKey:word];
	}
	
	[_dictionary removeAllObjects];
	_dictionary= newDictionary;
	
	_idfWeightsDirty= YES;
}


@end

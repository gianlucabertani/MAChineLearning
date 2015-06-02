//
//  MLWordDictionary.m
//  MAChineLearning
//
//  Created by Gianluca Bertani on 10/05/15.
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

#import "MLWordDictionary.h"
#import "MLWordInfo.h"
#import "MLBagOfWordsException.h"

#import "MLConstants.h"

#import <Accelerate/Accelerate.h>


#pragma -
#pragma WordDictionary extension

@interface MLWordDictionary () {
	NSMutableDictionary *_dictionary;
	NSUInteger _maxSize;
	
	NSUInteger _totalWords;
	NSUInteger _totalDocuments;
	MLReal *_idfWeights;
	
	NSMutableSet *_documents;
}


@end


#pragma -
#pragma WordDictionary implementation

@implementation MLWordDictionary


#pragma mark -
#pragma mark Initialization

+ (MLWordDictionary *) dictionaryWithMaxSize:(NSUInteger)maxSize {
	return [[MLWordDictionary alloc] initWithMaxSize:maxSize];
}

- (instancetype) initWithMaxSize:(NSUInteger)maxSize {
	if ((self = [super init])) {
		
		// Initialization
		_dictionary= [[NSMutableDictionary alloc] initWithCapacity:maxSize];
		_maxSize= maxSize;
		
		_totalWords= 0;
		_totalDocuments= 0;
		_idfWeights= NULL;
		
		_documents= [[NSMutableSet alloc] init];
	}
	
	return self;
}

- (void) dealloc {
	if (_idfWeights) {
		free(_idfWeights);
		_idfWeights= NULL;
	}
}


#pragma mark -
#pragma mark Dictionary access and manipulation

- (BOOL) containsWord:(NSString *)word {
	NSString *lowercaseWord= [word lowercaseString];
	
	return ([_dictionary objectForKey:lowercaseWord] != nil);
}

- (MLWordInfo *) infoForWord:(NSString *)word {
	NSString *lowercaseWord= [word lowercaseString];
	
	return [_dictionary objectForKey:lowercaseWord];
}

- (MLWordInfo *) addOccurrenceForWord:(NSString *)word textID:(NSString *)textID {
	NSString *lowercaseWord= [word lowercaseString];
	
	MLWordInfo *wordInfo= [_dictionary objectForKey:lowercaseWord];
	if ((!wordInfo) && (_dictionary.count < _maxSize)) {
		wordInfo= [[MLWordInfo alloc] initWithWord:word position:_dictionary.count];
		[_dictionary setObject:wordInfo forKey:lowercaseWord];
	}
	
	[wordInfo addOccurrenceForTextID:textID];
	
	if (textID && (![_documents containsObject:textID])) {
		[_documents addObject:textID];
		_totalDocuments++;
	}
	
	_totalWords++;
	
	return wordInfo;
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
}


#pragma mark -
#pragma mark Inverse document frequency

- (void) computeIDFWeights {
	if (_idfWeights) {
		free(_idfWeights);
		_idfWeights= NULL;
	}
	
	int err= posix_memalign((void **) &_idfWeights,
							BUFFER_MEMORY_ALIGNMENT,
							sizeof(MLReal) * _dictionary.count);
	if (err)
		@throw [MLBagOfWordsException bagOfWordsExceptionWithReason:@"Error while allocating buffer"
														 userInfo:@{@"buffer": @"idfWeights",
																	@"error": [NSNumber numberWithInt:err]}];
	
	// Clear the IDF buffer
	ML_VDSP_VCLR(_idfWeights, 1, _dictionary.count);
	
	// Compute inverse document frequency
	for (MLWordInfo *wordInfo in _dictionary) {
		MLReal weight= log(((MLReal) _totalDocuments) / (1.0 + ((MLReal) wordInfo.documentOccurrencies)));
		_idfWeights[wordInfo.position]= weight;
	}
}


#pragma mark -
#pragma mark NSObject overrides

- (NSString *) description {
	NSMutableString *descr= [[NSMutableString alloc] initWithCapacity:100 *_dictionary.count];
	
	[descr appendString:@"{\n"];

	NSArray *wordInfos= [_dictionary allValues];
	for (MLWordInfo *wordInfo in wordInfos)
		[descr appendFormat:@"\t'%@': %lu (%lu)\n", wordInfo.word, wordInfo.totalOccurrencies, wordInfo.documentOccurrencies];
	
	[descr appendString:@"}"];

	return [descr description];
}


#pragma mark -
#pragma mark Properties

@dynamic size;

- (NSUInteger) size {
	return _dictionary.count;
}

@synthesize maxSize= _maxSize;

@synthesize totalWords= _totalWords;
@synthesize totalDocuments= _totalDocuments;
@synthesize idfWeights= _idfWeights;


@end


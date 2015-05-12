//
//  WordDictionary.m
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

#import "WordDictionary.h"
#import "WordInfo.h"
#import "BagOfWordsException.h"

#import "Constants.h"

#import <Accelerate/Accelerate.h>


#pragma -
#pragma WordDictionary extension

@interface WordDictionary () {
	NSMutableDictionary *_dictionary;
	NSUInteger _maxSize;
	
	NSUInteger _totalWords;
	NSUInteger _totalDocuments;
	REAL *_idfWeights;
	
	NSMutableSet *_documents;
}


@end


#pragma -
#pragma WordDictionary implementation

@implementation WordDictionary


#pragma mark -
#pragma mark Initialization

+ (WordDictionary *) dictionaryWithMaxSize:(NSUInteger)maxSize {
	return [[WordDictionary alloc] initWithMaxSize:maxSize];
}

- (id) initWithMaxSize:(NSUInteger)maxSize {
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

- (WordInfo *) infoForWord:(NSString *)word {
	NSString *lowercaseWord= [word lowercaseString];
	
	return [_dictionary objectForKey:lowercaseWord];
}

- (WordInfo *) addOccurrenceForWord:(NSString *)word textID:(NSString *)textID {
	NSString *lowercaseWord= [word lowercaseString];
	
	WordInfo *wordInfo= [_dictionary objectForKey:lowercaseWord];
	if ((!wordInfo) && (_dictionary.count < _maxSize)) {
		wordInfo= [[WordInfo alloc] initWithPosition:_dictionary.count];
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

- (void) discardWordsWithOccurrenciesLessThan:(NSUInteger)minOccurrencies {
	[self applyFilter:^WordFilterOutcome(NSString *word, WordInfo *wordInfo) {
		return (wordInfo.totalOccurrencies < minOccurrencies) ? WordFilterOutcomeDiscardWord : WordFilterOutcomeKeepWord;
	}];
}

- (void) discardWordsWithOccurrenciesGreaterThan:(NSUInteger)maxOccurrencies {
	[self applyFilter:^WordFilterOutcome(NSString *word, WordInfo *wordInfo) {
		return (wordInfo.totalOccurrencies > maxOccurrencies) ? WordFilterOutcomeDiscardWord : WordFilterOutcomeKeepWord;
	}];
}

- (void) applyFilter:(WordFilter)filter {
	NSArray *words= [_dictionary allKeys];
	
	for (NSString *word in words) {
		WordInfo *wordInfo= [_dictionary objectForKey:word];

		WordFilterOutcome outcome= filter(word, wordInfo);
		switch (outcome) {
			case WordFilterOutcomeDiscardWord:
				[_dictionary removeObjectForKey:word];
				break;
				
			case WordFilterOutcomeKeepWord:
				break;
		}
	}
}

- (void) compact {
	NSMutableDictionary *newDictionary= [[NSMutableDictionary alloc] initWithCapacity:_dictionary.count];
	
	NSArray *words= [_dictionary allKeys];
	
	for (NSString *word in words) {
		WordInfo *wordInfo= [_dictionary objectForKey:word];

		WordInfo *newWordInfo= [[WordInfo alloc] initWithWordInfo:wordInfo newPosition:newDictionary.count];
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
							sizeof(REAL) * _dictionary.count);
	if (err)
		@throw [BagOfWordsException bagOfWordsExceptionWithReason:@"Error while allocating buffer"
														 userInfo:@{@"buffer": @"idfWeights",
																	@"error": [NSNumber numberWithInt:err]}];
	
	// Clear the IDF buffer
	nnVDSP_VCLR(_idfWeights, 1, _dictionary.count);
	
	// Compute inverse document frequency
	for (WordInfo *wordInfo in _dictionary) {
		REAL weight= log(((REAL) _totalDocuments) / (1.0 + ((REAL) wordInfo.documentOccurrencies)));
		_idfWeights[wordInfo.position]= weight;
	}
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


//
//  TokenDictionary.m
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

#import "TokenDictionary.h"
#import "TokenInfo.h"
#import "BagOfWordsException.h"

#import "Constants.h"

#import <Accelerate/Accelerate.h>


#pragma -
#pragma TokenDictionary extension

@interface TokenDictionary () {
	NSMutableDictionary *_dictionary;
	NSUInteger _maxSize;
	
	NSUInteger _totalDocuments;
	REAL *_idfWeights;
	
	NSMutableSet *_documents;
}


@end


#pragma -
#pragma TokenDictionary implementation

@implementation TokenDictionary


#pragma mark -
#pragma mark Initialization

+ (TokenDictionary *) dictionaryWithMaxSize:(NSUInteger)maxSize {
	return [[TokenDictionary alloc] initWithMaxSize:maxSize];
}

- (id) initWithMaxSize:(NSUInteger)maxSize {
	if ((self = [super init])) {
		
		// Initialization
		_dictionary= [[NSMutableDictionary alloc] initWithCapacity:maxSize];
		_maxSize= maxSize;
		
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

- (TokenInfo *) infoForToken:(NSString *)token {
	NSString *lowercaseToken= [token lowercaseString];
	
	return [_dictionary objectForKey:lowercaseToken];
}

- (TokenInfo *) addOccurrenceForToken:(NSString *)token textID:(NSString *)textID {
	NSString *lowercaseToken= [token lowercaseString];
	
	TokenInfo *tokenInfo= [_dictionary objectForKey:lowercaseToken];
	if ((!tokenInfo) && (_dictionary.count < _maxSize)) {
		tokenInfo= [[TokenInfo alloc] initWithPosition:_dictionary.count];
		[_dictionary setObject:tokenInfo forKey:lowercaseToken];
	}
	
	[tokenInfo addOccurrenceForTextID:textID];
	
	if (textID && (![_documents containsObject:textID])) {
		[_documents addObject:textID];
		_totalDocuments++;
	}
	
	return tokenInfo;
}


#pragma mark -
#pragma mark Dictionary filtering

- (void) discardTokensWithOccurrenciesLessThan:(NSUInteger)minOccurrencies {
	[self applyFilter:^TokenFilterOutcome(NSString *token, TokenInfo *tokenInfo) {
		return (tokenInfo.totalOccurrencies < minOccurrencies) ? TokenFilterOutcomeDiscardToken : TokenFilterOutcomeKeepToken;
	}];
}

- (void) discardTokensWithOccurrenciesGreaterThan:(NSUInteger)maxOccurrencies {
	[self applyFilter:^TokenFilterOutcome(NSString *token, TokenInfo *tokenInfo) {
		return (tokenInfo.totalOccurrencies > maxOccurrencies) ? TokenFilterOutcomeDiscardToken : TokenFilterOutcomeKeepToken;
	}];
}

- (void) applyFilter:(TokenFilter)filter {
	NSArray *tokens= [_dictionary allKeys];
	
	for (NSString *token in tokens) {
		TokenInfo *tokenInfo= [_dictionary objectForKey:token];

		TokenFilterOutcome outcome= filter(token, tokenInfo);
		switch (outcome) {
			case TokenFilterOutcomeDiscardToken:
				[_dictionary removeObjectForKey:token];
				break;
				
			case TokenFilterOutcomeKeepToken:
				break;
		}
	}
}

- (void) compact {
	NSMutableDictionary *newDictionary= [[NSMutableDictionary alloc] initWithCapacity:_dictionary.count];
	
	NSArray *tokens= [_dictionary allKeys];
	
	for (NSString *token in tokens) {
		TokenInfo *tokenInfo= [_dictionary objectForKey:token];

		TokenInfo *newTokenInfo= [[TokenInfo alloc] initWithTokenInfo:tokenInfo newPosition:newDictionary.count];
		[newDictionary setObject:newTokenInfo forKey:token];
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
							sizeof(REAL) * _maxSize);
	if (err)
		@throw [BagOfWordsException bagOfWordsExceptionWithReason:@"Error while allocating buffer"
														 userInfo:@{@"buffer": @"idfWeights",
																	@"error": [NSNumber numberWithInt:err]}];
	
	// Clear the IDF buffer
	nnVDSP_VCLR(_idfWeights, 1, _maxSize);
	
	// Compute inverse document frequency
	for (TokenInfo *tokenInfo in _dictionary) {
		REAL weight= log(((REAL) _totalDocuments) / (1.0 + ((REAL) tokenInfo.documentOccurrencies)));
		_idfWeights[tokenInfo.position]= weight;
	}
}


#pragma mark -
#pragma mark Properties

@dynamic size;

- (NSUInteger) size {
	return _dictionary.count;
}

@synthesize maxSize= _maxSize;

@synthesize totalDocuments= _totalDocuments;
@synthesize idfWeights= _idfWeights;


@end


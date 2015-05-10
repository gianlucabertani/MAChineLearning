//
//  TokenInfo.m
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

#import "TokenInfo.h"


#pragma -
#pragma TokenInfo extension

@interface TokenInfo () {
	NSUInteger _position;
	NSUInteger _totalOccurrencies;
	NSUInteger _documentOccurrencies;
	
	NSMutableSet *_documents;
}


#pragma -
#pragma Internal properties

@property (nonatomic, readonly) NSSet *documents;


@end


#pragma -
#pragma TokenInfo extension

@implementation TokenInfo


#pragma -
#pragma Initialization

- (id) initWithTokenInfo:(TokenInfo *)tokenInfo newPosition:(NSUInteger)newPosition {
	if ((self = [super init])) {
		
		// Initialization
		_position= newPosition;
		_totalOccurrencies= tokenInfo.totalOccurrencies;
		_documentOccurrencies= tokenInfo.documentOccurrencies;
		
		_documents= [[NSMutableSet alloc] initWithSet:tokenInfo.documents];
	}
	
	return self;
}

- (id) initWithPosition:(NSUInteger)position {
	if ((self = [super init])) {
		
		// Initialization
		_position= position;
		_totalOccurrencies= 0;
		_documentOccurrencies= 0;
		
		_documents= [[NSMutableSet alloc] init];
	}
	
	return self;
}


#pragma -
#pragma Occurrencies counting

- (void) addOccurrenceForTextID:(NSString *)textID {
	_totalOccurrencies++;
	
	if (textID && (![_documents containsObject:textID])) {
		[_documents addObject:textID];
		_documentOccurrencies++;
	}
}


#pragma -
#pragma Properties

@synthesize position= _position;
@synthesize totalOccurrencies= _totalOccurrencies;
@synthesize documentOccurrencies= _documentOccurrencies;

@synthesize documents= _documents;


@end

//
//  MLTextFragment.m
//  MAChineLearning
//
//  Created by Gianluca Bertani on 26/04/15.
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

#import "MLTextFragment.h"
#import "MLBagOfWordsException.h"


#pragma -
#pragma TextFragment extension

@interface MLTextFragment () {
	NSString *_fragment;
	NSRange _range;
	NSRange _sentenceRange;
	float _tokenIndex;
	NSString *_linguisticTag;
}


@end


#pragma -
#pragma TextFragment implementation

@implementation MLTextFragment


#pragma -
#pragma Initialization

- (instancetype) initWithFrament:(NSString *)fragment range:(NSRange)range sentenceRange:(NSRange)sentenceRange tokenIndex:(float)index {
	if ((self = [super init])) {
		
		// Initialization
		_fragment= fragment;
		_range= range;
		_sentenceRange= sentenceRange;
		_tokenIndex= index;
	}
	
	return self;
}

- (instancetype) initWithFrament:(NSString *)fragment range:(NSRange)range sentenceRange:(NSRange)sentenceRange tokenIndex:(float)index linguisticTag:(NSString *)linguisticTag {
	if ((self = [super init])) {
		
		// Initialization
		_fragment= fragment;
		_range= range;
		_sentenceRange= sentenceRange;
		_tokenIndex= index;
		_linguisticTag= linguisticTag;
	}
	
	return self;
}


#pragma -
#pragma Continuity check and combination

- (BOOL) isContiguous:(MLTextFragment *)previousFragment {
	return ((previousFragment.tokenIndex == _tokenIndex -1.0) &&
			(previousFragment.sentenceRange.location == _sentenceRange.location));
}

- (MLTextFragment *) combineWithFragment:(MLTextFragment *)previousFragment {
	NSString *combinedText= [NSString stringWithFormat:@"%@ %@", previousFragment.fragment, _fragment];
	NSRange combinedRange= NSMakeRange(previousFragment.range.location, (_range.location + _range.length) - previousFragment.range.location);
	float combinedTokenIndex= _tokenIndex + 0.1 + (previousFragment.tokenIndex - ((int) previousFragment.tokenIndex));

	MLTextFragment *combinedFragment= [[MLTextFragment alloc] initWithFrament:combinedText
																	range:combinedRange
															sentenceRange:_sentenceRange
															   tokenIndex:combinedTokenIndex];

	return combinedFragment;
}


#pragma -
#pragma NSObject overrides

- (BOOL) isEqual:(id)object {
	if (![object isKindOfClass:[MLTextFragment class]])
		@throw [MLBagOfWordsException bagOfWordsExceptionWithReason:@"Trying to compare an MLTextFragment with something else"
														 userInfo:@{@"self": self,
																	@"object": object}];
	
	MLTextFragment *fragment= (MLTextFragment *) object;
	return ([_fragment isEqualToString:fragment.fragment] &&
			(_range.location == fragment.range.location));
}

- (NSUInteger) hash {
    return (_fragment.hash + _range.location);
}

- (NSString *) description {
	return [NSString stringWithFormat:@"<%.2f '%@' %lu:%lu>", _tokenIndex, _fragment, _range.location, _range.length];
}


#pragma -
#pragma Properties

@synthesize fragment= _fragment;
@synthesize range= _range;
@synthesize sentenceRange= _sentenceRange;
@synthesize tokenIndex= _tokenIndex;
@synthesize linguisticTag= _linguisticTag;


@end

//
//  MLWordInfo.m
//  MAChineLearning
//
//  Created by Gianluca Bertani on 10/05/15.
//  Copyright (c) 2015-2018 Gianluca Bertani. All rights reserved.
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

#import "MLWordInfo.h"

#import "MLBagOfWordsException.h"


@implementation MLWordInfo


#pragma -
#pragma Initialization

- (instancetype) init {
    @throw [MLBagOfWordsException bagOfWordsExceptionWithReason:@"MLWordInfo class must be initialized properly"
                                                       userInfo:nil];
}

- (instancetype) initWithWord:(NSString *)word position:(NSUInteger)position {
    if ((self = [super init])) {
        
        // Initialization
        _word= word;
        _position= position;
        
        _totalOccurrencies= 0;
        _documentOccurrencies= 0;
        
        _documentIDs= [[NSMutableSet alloc] init];
    }
    
    return self;
}

- (instancetype) initWithWordInfo:(MLWordInfo *)wordInfo newPosition:(NSUInteger)newPosition {
    if ((self = [super init])) {
        
        // Initialization
        _word= wordInfo.word;
        _position= newPosition;
        
        _totalOccurrencies= wordInfo.totalOccurrencies;
        _documentOccurrencies= wordInfo.documentOccurrencies;
        
        _documentIDs= [[NSMutableSet alloc] initWithSet:wordInfo.documentIDs];
    }
    
    return self;
}


#pragma -
#pragma Properties

@synthesize word= _word;
@synthesize position= _position;

@synthesize totalOccurrencies= _totalOccurrencies;
@synthesize documentOccurrencies= _documentOccurrencies;

@synthesize documentIDs= _documentIDs;


@end

//
//  MLLayer.m
//  MAChineLearning
//
//  Created by Gianluca Bertani on 01/03/15.
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

#import "MLLayer.h"

#import "MLNeuralNetworkException.h"


#pragma mark -
#pragma mark Layer extension

@interface MLLayer () {    
    MLLayer __weak *_previousLayer;
    MLLayer __weak *_nextLayer;
}


@end


#pragma mark -
#pragma mark Layer implementation

@implementation MLLayer


#pragma mark -
#pragma mark Initialization

- (instancetype) init {
    @throw [MLNeuralNetworkException neuralNetworkExceptionWithReason:@"MLLayer class must be initialized properly"
                                                             userInfo:nil];
}

- (instancetype) initWithIndex:(NSUInteger)index size:(NSUInteger)size {
    if ((self = [super init])) {
        
        // Initialization
        _index= index;
        _size= size;
    }
    
    return self;
}


#pragma mark -
#pragma mark Setup

- (void) setUp {
    
    // Nothing to do
}


#pragma mark -
#pragma mark Properties

@synthesize index= _index;
@synthesize size= _size;

@synthesize previousLayer= _previousLayer;
@synthesize nextLayer= _nextLayer;


@end

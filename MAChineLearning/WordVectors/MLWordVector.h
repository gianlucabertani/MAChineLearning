//
//  MLWordVector.h
//  MAChineLearning
//
//  Created by Gianluca Bertani on 03/06/15.
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

#import <Foundation/Foundation.h>

#import "MLReal.h"


@class MLWordInfo;

@interface MLWordVector : NSObject


#pragma -
#pragma Initialization

- (nonnull instancetype) init NS_UNAVAILABLE;

- (nonnull instancetype) initWithVector:(nonnull MLReal *)vector
                                   size:(NSUInteger)size
                    freeVectorOnDealloc:(BOOL)freeOnDealloc
                                        NS_DESIGNATED_INITIALIZER;


#pragma -
#pragma Vector algebra and comparison

- (nonnull MLWordVector *) addVector:(nonnull MLWordVector *)vector;
- (nonnull MLWordVector *) subtractVector:(nonnull MLWordVector *)vector;

- (MLReal) similarityToVector:(nonnull MLWordVector *)vector;
- (MLReal) distanceToVector:(nonnull MLWordVector *)vector;


#pragma -
#pragma Properties

@property (nonatomic, readonly, nonnull) MLReal *vector;
@property (nonatomic, readonly) NSUInteger size;
@property (nonatomic, readonly) MLReal magnitude;


@end

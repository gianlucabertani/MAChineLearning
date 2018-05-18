//
//  MLTextFragment.h
//  MAChineLearning
//
//  Created by Gianluca Bertani on 26/04/15.
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


@interface MLTextFragment : NSObject


#pragma -
#pragma Initialization

- (nonnull instancetype) initWithFrament:(nonnull NSString *)fragment
                                   range:(NSRange)range
                           sentenceRange:(NSRange)sentenceRange
                              tokenIndex:(float)index;

- (nonnull instancetype) initWithFrament:(nonnull NSString *)fragment
                                   range:(NSRange)range
                           sentenceRange:(NSRange)sentenceRange
                              tokenIndex:(float)index
                           linguisticTag:(nullable NSString *)linguisticTag;


#pragma -
#pragma Continuity check and combination

- (BOOL) isContiguous:(nonnull MLTextFragment *)previousFragment;

- (nonnull MLTextFragment *) combineWithFragment:(nonnull MLTextFragment *)previousFragment;


#pragma -
#pragma Properties

@property (nonatomic, readonly, nonnull) NSString *fragment;
@property (nonatomic, readonly) NSRange range;
@property (nonatomic, readonly) NSRange sentenceRange;
@property (nonatomic, readonly) float tokenIndex;
@property (nonatomic, readonly, nullable) NSString *linguisticTag;


@end

//
//  NSString+WordUtils.m
//  MAChineLearning
//
//  Created by Gianluca Bertani on 23/04/15.
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

#import "NSString+WordUtils.h"


#pragma mark -
#pragma mark NSString word utilities

@implementation NSString (WordUtils)


#pragma mark -
#pragma mark Capitalization checks

- (BOOL) isCapitalizedString {
	return (([self length] >= 2) &&
			[[NSCharacterSet uppercaseLetterCharacterSet] characterIsMember:[self characterAtIndex:0]] &&
			[[NSCharacterSet lowercaseLetterCharacterSet] characterIsMember:[self characterAtIndex:1]]);
}

- (BOOL) isAcronym {
	return (([self length] >= 2) &&
			([self length] <= 7) &&
			[[self uppercaseString] isEqualToString:self]);
}

- (BOOL) isNameInitial {
	return ((([self length] == 1) && [[NSCharacterSet uppercaseLetterCharacterSet] characterIsMember:[self characterAtIndex:0]]) ||
			(([self length] == 2) && [[NSCharacterSet uppercaseLetterCharacterSet] characterIsMember:[self characterAtIndex:0]] &&
			 [[NSCharacterSet uppercaseLetterCharacterSet] characterIsMember:[self characterAtIndex:1]]));
}


@end

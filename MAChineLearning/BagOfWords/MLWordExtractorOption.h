//
//  MLWordExtractorOption.h
//  MAChineLearning
//
//  Created by Gianluca Bertani on 23/04/15.
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

#ifndef MAChineLearning_MLWordExtractorOption_h
#define MAChineLearning_MLWordExtractorOption_h


typedef enum {
	MLWordExtractorOptionOmitStopWords= 1,

	MLWordExtractorOptionOmitVerbs= 1 << 1,
	MLWordExtractorOptionOmitAdjectives= 1 << 2,
	MLWordExtractorOptionOmitAdverbs= 1 << 3,
	MLWordExtractorOptionOmitNouns= 1 << 4,
	MLWordExtractorOptionOmitNames= 1 << 5,
	MLWordExtractorOptionOmitNumbers= 1 << 6,
	MLWordExtractorOptionOmitOthers= 1 << 7,

	MLWordExtractorOptionKeepVerbAdjectiveCombos= 1 << 10,
	MLWordExtractorOptionKeepAdjectiveNounCombos= 1 << 11,
	MLWordExtractorOptionKeepAdverbNounCombos= 1 << 12,
	MLWordExtractorOptionKeepNounNounCombos= 1 << 13,
	MLWordExtractorOptionKeepNounVerbCombos= 1 << 14,

	MLWordExtractorOptionKeep2WordNames= 1 << 16,
	MLWordExtractorOptionKeep3WordNames= 1 << 17,
	
	MLWordExtractorOptionKeepAllBigrams= 1 << 20,
	MLWordExtractorOptionKeepAllTrigrams= 1 << 21,
	
	MLWordExtractorOptionKeepEmoticons= 1 << 24,
	
} MLWordExtractorOption;


#endif

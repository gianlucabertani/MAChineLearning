//
//  BagOfWords.m
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

#import "BagOfWords.h"
#import "TokenDictionary.h"
#import "TokenInfo.h"
#import "TextFragment.h"
#import "StopWords.h"
#import "NSString+WordUtils.h"
#import "BagOfWordsException.h"

#import "Constants.h"

#import <Accelerate/Accelerate.h>

#define LEFT_TO_RIGHT_EMOTICON             (@"\\s[:=;B]-?[)(|\\/\\\\\\]\\[DOoPp]")
#define RIGHT_TO_LEFT_EMOTICON             (@"\\s[)(|\\/\\\\\\]\\[DOo]-?[:=;]")
#define EMOJI                              (@"[😀😁😂😃😄😅😆😇😊😉👿😈☺️😋😌😍😑😐😏😎😒😓😔😕😙😘😗😖😚😛😜😝😡😠😟😞😢😣😤😥😩😨😧😦😪😫😬😭😱😰😯😮😲😳😴😵😶😷]")

#define GUESS_THRESHOLD_PERC              (10)


#pragma mark -
#pragma mark BagOfWords extension

@interface BagOfWords () {
	NSString *_textID;
	NSArray *_tokens;

	NSUInteger _outputSize;
	REAL *_outputBuffer;
	BOOL _localBuffer;
}


#pragma mark -
#pragma mark Extractor support

+ (void) insertEmoticonFragments:(NSString *)text fragments:(NSMutableArray *)fragments;


#pragma mark -
#pragma mark Internals

- (void) prepareOutputBuffer:(REAL *)outputBuffer;
- (void) fillOutputBuffer:(TokenDictionary *)dictionary buildDictionary:(BOOL)buildDictionary featureNormalization:(FeatureNormalizationType)normalizationType;
- (void) normalizeOutputBuffer:(TokenDictionary *)dictionary featureNormalization:(FeatureNormalizationType)normalizationType;


@end


#pragma mark -
#pragma mark BagOfWords statics

static NSDictionary *__stopWords= nil;


#pragma mark -
#pragma mark BagOfWords implementation

@implementation BagOfWords


#pragma mark -
#pragma mark Initialization

+ (BagOfWords *) bagOfWordsForTopicClassificationWithText:(NSString *)text textID:(NSString *)textID dictionary:(TokenDictionary *)dictionary language:(NSString *)languageCode featureNormalization:(FeatureNormalizationType)normalizationType {
	return [BagOfWords bagOfWordsWithText:text
								   textID:textID
							   dictionary:dictionary
						  buildDictionary:YES
								 language:languageCode
							wordExtractor:WordExtractorTypeLinguisticTagger
						 extractorOptions:WordExtractorOptionOmitStopWords | WordExtractorOptionOmitVerbs | WordExtractorOptionOmitAdjectives | WordExtractorOptionOmitAdverbs | WordExtractorOptionOmitNouns | WordExtractorOptionOmitOthers | WordExtractorOptionKeepAdjectiveNounCombos | WordExtractorOptionKeepAdverbNounCombos | WordExtractorOptionKeepNounNounCombos | WordExtractorOptionKeep2WordNames | WordExtractorOptionKeep3WordNames
					 featureNormalization:normalizationType
							 outputBuffer:nil];
}

+ (BagOfWords *) bagOfWordsForSentimentAnalysisWithText:(NSString *)text textID:(NSString *)textID dictionary:(TokenDictionary *)dictionary language:(NSString *)languageCode featureNormalization:(FeatureNormalizationType)normalizationType {
	return [BagOfWords bagOfWordsWithText:text
								   textID:textID
							   dictionary:dictionary
						  buildDictionary:YES
								 language:languageCode
							wordExtractor:WordExtractorTypeSimpleTokenizer
						 extractorOptions:WordExtractorOptionOmitStopWords | WordExtractorOptionKeepEmoticons | WordExtractorOptionKeepAllBigrams
					 featureNormalization:normalizationType
							 outputBuffer:nil];
}

+ (BagOfWords *) bagOfWordsWithText:(NSString *)text textID:(NSString *)textID dictionary:(TokenDictionary *)dictionary buildDictionary:(BOOL)buildDictionary language:(NSString *)languageCode wordExtractor:(WordExtractorType)extractorType extractorOptions:(WordExtractorOption)extractorOptions featureNormalization:(FeatureNormalizationType)normalizationType outputBuffer:(REAL *)outputBuffer {
	BagOfWords *bagOfWords= [[BagOfWords alloc] initWithText:text
													  textID:textID
												  dictionary:dictionary
											 buildDictionary:buildDictionary
													language:languageCode
											   wordExtractor:extractorType
											extractorOptions:extractorOptions
										featureNormalization:normalizationType
												outputBuffer:outputBuffer];
	
	return bagOfWords;
}

+ (BagOfWords *) bagOfWordsWithTokens:(NSArray *)tokens textID:(NSString *)textID dictionary:(TokenDictionary *)dictionary buildDictionary:(BOOL)buildDictionary featureNormalization:(FeatureNormalizationType)normalizationType outputBuffer:(REAL *)outputBuffer {
	BagOfWords *bagOfWords= [[BagOfWords alloc] initWithTokens:tokens
														textID:textID
													dictionary:dictionary
											   buildDictionary:buildDictionary
										  featureNormalization:normalizationType
												  outputBuffer:outputBuffer];
	
	return bagOfWords;
}

- (id) initWithText:(NSString *)text textID:(NSString *)textID dictionary:(TokenDictionary *)dictionary buildDictionary:(BOOL)buildDictionary language:(NSString *)languageCode wordExtractor:(WordExtractorType)extractorType extractorOptions:(WordExtractorOption)extractorOptions featureNormalization:(FeatureNormalizationType)normalizationType outputBuffer:(REAL *)outputBuffer {
	if ((self = [super init])) {
		
		// Fill stop words if not filled already
		if (!__stopWords)
			__stopWords= STOP_WORDS;
		
		// Checks
		if (!dictionary)
			@throw [BagOfWordsException bagOfWordsExceptionWithReason:@"Missing dictionary"
															 userInfo:nil];
		
		if ((extractorOptions & WordExtractorOptionOmitStopWords) &&
			(!languageCode))
			@throw [BagOfWordsException bagOfWordsExceptionWithReason:@"Missing language code (language is needed to skip stopwords)"
															 userInfo:nil];
		
		if (((extractorOptions & WordExtractorOptionOmitVerbs) |
			 (extractorOptions & WordExtractorOptionOmitAdjectives) |
			 (extractorOptions & WordExtractorOptionOmitAdverbs) |
			 (extractorOptions & WordExtractorOptionOmitNouns) |
			 (extractorOptions & WordExtractorOptionOmitNames) |
			 (extractorOptions & WordExtractorOptionOmitOthers) |
			 (extractorOptions & WordExtractorOptionKeepAdjectiveNounCombos) |
			 (extractorOptions & WordExtractorOptionKeepAdverbNounCombos) |
			 (extractorOptions & WordExtractorOptionKeepNounNounCombos) |
			 (extractorOptions & WordExtractorOptionKeepNounVerbCombos) |
			 (extractorOptions & WordExtractorOptionKeepVerbAdjectiveCombos) |
			 (extractorOptions & WordExtractorOptionKeep2WordNames) |
			 (extractorOptions & WordExtractorOptionKeep3WordNames)) &&
			(extractorType != WordExtractorTypeLinguisticTagger))
			@throw [BagOfWordsException bagOfWordsExceptionWithReason:@"Options on verbs, adjectives, adverbs, nouns and names require the linguistic tagger"
															 userInfo:nil];
		
		switch (normalizationType) {
			case FeatureNormalizationTypeL2TFiDF:
				if (buildDictionary)
					@throw [BagOfWordsException bagOfWordsExceptionWithReason:@"TF-iDF normalization requires a pre-built dictionary"
																	 userInfo:nil];
				
			default:
				break;
		}
		
		// Initialization
		_textID= textID;
		
		// Run the appropriate extractor
		switch (extractorType) {
			case WordExtractorTypeSimpleTokenizer:
				_tokens= [BagOfWords extractTokensWithSimpleTokenizer:text language:languageCode extractorOptions:extractorOptions];
				break;
				
			case WordExtractorTypeLinguisticTagger:
				_tokens= [BagOfWords extractTokensWithLinguisticTagger:text language:languageCode extractorOptions:extractorOptions];
				break;
		}
		
		_outputSize= (buildDictionary ? dictionary.maxSize : dictionary.size);
		
		// Set up the output buffer
		[self prepareOutputBuffer:outputBuffer];
		
		// Build dictionary and the output buffer
		[self fillOutputBuffer:dictionary buildDictionary:buildDictionary featureNormalization:normalizationType];

		// Apply vector-wide normalization
		[self normalizeOutputBuffer:dictionary featureNormalization:normalizationType];
	}
	
	return self;
}

- (id) initWithTokens:(NSArray *)tokens textID:(NSString *)textID dictionary:(TokenDictionary *)dictionary buildDictionary:(BOOL)buildDictionary featureNormalization:(FeatureNormalizationType)normalizationType outputBuffer:(REAL *)outputBuffer {
	if ((self = [super init])) {
		
		// Checks
		if (!dictionary)
			@throw [BagOfWordsException bagOfWordsExceptionWithReason:@"Missing dictionary"
															 userInfo:nil];
		
		switch (normalizationType) {
			case FeatureNormalizationTypeL2TFiDF:
				if (buildDictionary)
					@throw [BagOfWordsException bagOfWordsExceptionWithReason:@"TF-iDF normalization requires a pre-built dictionary"
																	 userInfo:nil];
				
			default:
				break;
		}
	
		// Initialization
		_textID= textID;
		_tokens= tokens;

		_outputSize= (buildDictionary ? dictionary.maxSize : dictionary.size);

		// Set up the output buffer
		[self prepareOutputBuffer:outputBuffer];
		
		// Build dictionary and the output buffer
		[self fillOutputBuffer:dictionary buildDictionary:buildDictionary featureNormalization:normalizationType];

		// Apply vector-wide normalization
		[self normalizeOutputBuffer:dictionary featureNormalization:normalizationType];
	}
	
	return self;
}

- (void) dealloc {
	if (_localBuffer) {
		free(_outputBuffer);
		_outputBuffer= NULL;
	}
}


#pragma mark -
#pragma mark Internals

- (void) prepareOutputBuffer:(REAL *)outputBuffer {
	if (outputBuffer) {
		_outputBuffer= outputBuffer;
		_localBuffer= NO;
		
	} else {
		int err= posix_memalign((void **) &_outputBuffer,
								BUFFER_MEMORY_ALIGNMENT,
								sizeof(REAL) * _outputSize);
		if (err)
			@throw [BagOfWordsException bagOfWordsExceptionWithReason:@"Error while allocating buffer"
															 userInfo:@{@"buffer": @"outputBuffer",
																		@"error": [NSNumber numberWithInt:err]}];
		
		_localBuffer= YES;
	}
	
	// Clear the output buffer
	nnVDSP_VCLR(_outputBuffer, 1, _outputSize);
}

- (void) fillOutputBuffer:(TokenDictionary *)dictionary buildDictionary:(BOOL)buildDictionary featureNormalization:(FeatureNormalizationType)normalizationType {
	
	// Build dictionary and the output buffer
	for (NSString *token in _tokens) {
		TokenInfo *tokenInfo= nil;
		
		if (buildDictionary)
			tokenInfo= [dictionary addOccurrenceForToken:token textID:_textID];
		else
			tokenInfo= [dictionary infoForToken:token];
		
		if (tokenInfo) {
			switch (normalizationType) {
				case FeatureNormalizationTypeNone:
				case FeatureNormalizationTypeL2:
				case FeatureNormalizationTypeL2TFiDF:
					_outputBuffer[tokenInfo.position] += 1.0;
					break;
					
				case FeatureNormalizationTypeBoolean:
					_outputBuffer[tokenInfo.position]= 1.0;
					break;
			}
		}
	}
}

- (void) normalizeOutputBuffer:(TokenDictionary *)dictionary featureNormalization:(FeatureNormalizationType)normalizationType {
	
	// Apply vector-wide normalization
	switch (normalizationType) {
		case FeatureNormalizationTypeL2TFiDF: {
			if (!dictionary.idfWeights)
				[dictionary computeIDFWeights];
			
			// Multiply by IDF weights
			nnVDSP_VMUL(_outputBuffer, 1, dictionary.idfWeights, 1, _outputBuffer, 1, _outputSize);
			
			// NOTE: No "break" intended here
		}
			
		case FeatureNormalizationTypeL2: {
			REAL normL2= 0.0;
			
			for (NSString *token in _tokens) {
				TokenInfo *tokenInfo= [dictionary infoForToken:token];
				normL2 += _outputBuffer[tokenInfo.position] * _outputBuffer[tokenInfo.position];
			}
			
			normL2= sqrt(normL2);
			nnVDSP_VSDIV(_outputBuffer, 1, &normL2, _outputBuffer, 1, _outputSize);
			break;
		}
			
		default:
			break;
	}
}


#pragma mark -
#pragma mark Dictionary building

+ (void) buildDictionaryWithText:(NSString *)text textID:(NSString *)textID dictionary:(TokenDictionary *)dictionary language:(NSString *)languageCode wordExtractor:(WordExtractorType)extractorType extractorOptions:(WordExtractorOption)extractorOptions {
	NSArray *tokens= nil;
	
	// Run the appropriate word extractor
	switch (extractorType) {
		case WordExtractorTypeSimpleTokenizer:
			tokens= [BagOfWords extractTokensWithSimpleTokenizer:text language:languageCode extractorOptions:extractorOptions];
			break;
			
		case WordExtractorTypeLinguisticTagger:
			tokens= [BagOfWords extractTokensWithLinguisticTagger:text language:languageCode extractorOptions:extractorOptions];
			break;
	}
	
	for (NSString *token in tokens)
		[dictionary addOccurrenceForToken:token textID:textID];
}


#pragma mark -
#pragma mark Languages code guessing

+ (NSString *) guessLanguageCodeWithLinguisticTagger:(NSString *)text {
	@autoreleasepool {
	
		// Init the linguistic tagger
		NSLinguisticTagger *tagger= [[NSLinguisticTagger alloc] initWithTagSchemes:@[NSLinguisticTagSchemeLanguage] options:0];
		[tagger setString:text];
	
		// Get the language using the tagger
		NSString *language= [tagger tagAtIndex:0 scheme:NSLinguisticTagSchemeLanguage tokenRange:NULL sentenceRange:NULL];
		return language;
	}
}

+ (NSString *) guessLanguageCodeWithStopWords:(NSString *)text {
	@autoreleasepool {
	
		// Fill stop words if not filled already
		if (!__stopWords)
			__stopWords= STOP_WORDS;
		
		// Prepare the score table
		NSMutableDictionary *scores= [NSMutableDictionary dictionary];
		
		// Search for stopwords in text, each occurrence counts as 1
		int wordsCount= 0;
		NSArray *words= [text componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
		for (NSString *word in words) {
			NSString *trimmedWord= [word stringByTrimmingCharactersInSet:[NSCharacterSet punctuationCharacterSet]];
			
			// Skip empty words
			if (trimmedWord.length == 0)
				continue;

			NSString *lowerCaseWord= [trimmedWord lowercaseString];
			wordsCount++;
			
			for (NSString *language in [__stopWords allKeys]) {
				NSSet *stopwords= [__stopWords objectForKey:language];
				
				if ([stopwords containsObject:lowerCaseWord]) {
					int score= [[scores objectForKey:language] intValue];
					[scores setObject:[NSNumber numberWithInt:score +1] forKey:language];
				}
			}
		}
		
		if (wordsCount == 0)
			return nil;
		
		// Remove languages below guess threshold
		for (NSString *language in [__stopWords allKeys]) {
			int score= [[scores objectForKey:language] intValue];
			int perc= (100 * score) / wordsCount;

			if (perc < GUESS_THRESHOLD_PERC)
				[scores removeObjectForKey:language];
		}
		
		// Check results
		if (scores.count == 0)
			return nil;
		
		if (scores.count == 1)
			return [[scores allKeys] firstObject];
		
		// Sort languages by scores and take the highest one
		NSArray *sortedScores= [[scores allKeys] sortedArrayUsingComparator:^NSComparisonResult(id obj1, id obj2) {
			NSString *language1= (NSString *) obj1;
			NSString *language2= (NSString *) obj2;
			
			NSNumber *score1= [scores objectForKey:language1];
			NSNumber *score2= [scores objectForKey:language2];
			
			return [score1 compare:score2];
		}];
		
		return [sortedScores lastObject];
	}
}


#pragma mark -
#pragma mark Word extractors

+ (NSArray *) extractTokensWithLinguisticTagger:(NSString *)text language:(NSString *)languageCode extractorOptions:(WordExtractorOption)extractorOptions {
	@autoreleasepool {
	
		// Make sure full-stops and apostrophes are followed by spaces
		text= [text stringByReplacingOccurrencesOfString:@"." withString:@". "];
		text= [text stringByReplacingOccurrencesOfString:@"'" withString:@"' "];
		
		// Prepare containers and stopwords list
		NSMutableArray *fragments= [NSMutableArray arrayWithCapacity:text.length / 5];
		NSMutableArray *combinedFragments= [NSMutableArray arrayWithCapacity:text.length / 10];
		NSMutableArray *fragmentsToBeOmitted= [NSMutableArray arrayWithCapacity:text.length / 10];
		NSSet *stopWords= [__stopWords objectForKey:languageCode];
		
		// Scan text with the linguistic tagger
		NSLinguisticTagger *tagger= [[NSLinguisticTagger alloc] initWithTagSchemes:@[NSLinguisticTagSchemeLexicalClass] options:0];
		[tagger setString:text];
		
		__block int tokenIndex= -1;
		[tagger enumerateTagsInRange:NSMakeRange(0, [text length])
							  scheme:NSLinguisticTagSchemeLexicalClass
							 options:NSLinguisticTaggerOmitPunctuation | NSLinguisticTaggerOmitWhitespace | NSLinguisticTaggerOmitOther
						  usingBlock:^(NSString *tag, NSRange tokenRange, NSRange sentenceRange, BOOL *stop) {
							  NSString *word= [text substringWithRange:tokenRange];
							  
							  // Clean up residual punctuation
							  word= [word stringByTrimmingCharactersInSet:[NSCharacterSet punctuationCharacterSet]];
							  if (word.length < 2)
								  return;
							  
							  tokenIndex++;
							  
							  // Skip stopwords if requested
							  if ((extractorOptions & WordExtractorOptionOmitStopWords) &&
								  [stopWords containsObject:[word lowercaseString]])
								  return;

							  // Create the fragment
							  TextFragment *fragment= [[TextFragment alloc] initWithFrament:word
																					  range:tokenRange
																			  sentenceRange:sentenceRange
																				 tokenIndex:tokenIndex
																			  linguisticTag:tag];
							  
							  [fragments addObject:fragment];
							  
							  // Skip verbs if requested
							  if ((extractorOptions & WordExtractorOptionOmitVerbs) &&
								  (tag == NSLinguisticTagVerb)) {

								  // Store for later removal
								  [fragmentsToBeOmitted addObject:fragment];
							  }
							  
							  // Skip adjectives if requested
							  if ((extractorOptions & WordExtractorOptionOmitAdjectives) &&
								  (tag == NSLinguisticTagAdjective)) {
								  
								  // Store for later removal
								  [fragmentsToBeOmitted addObject:fragment];
							  }
							  
							  // Skip adverbs if requested
							  if ((extractorOptions & WordExtractorOptionOmitAdverbs) &&
								  (tag == NSLinguisticTagAdverb)) {
								  
								  // Store for later removal
								  [fragmentsToBeOmitted addObject:fragment];
							  }
							  
							  // Skip nouns if requested
							  if ((extractorOptions & WordExtractorOptionOmitNouns) &&
								  (tag == NSLinguisticTagNoun) &&
								  (!([word isCapitalizedString] || [word isNameInitial] || [word isAcronym]))) {
								  
								  // Store for later removal
								  [fragmentsToBeOmitted addObject:fragment];
							  }

							  // Skip proper names if requested
							  if ((extractorOptions & WordExtractorOptionOmitNames) &&
								  (tag == NSLinguisticTagNoun) &&
								  ([word isCapitalizedString] || [word isNameInitial] || [word isAcronym])) {
								  
								  // Store for later removal
								  [fragmentsToBeOmitted addObject:fragment];
							  }
							  
							  // Skip numbers if requested
							  if ((extractorOptions & WordExtractorOptionOmitNumbers) &&
								  (tag == NSLinguisticTagNumber)) {
								  
								  // Store for later removal
								  [fragmentsToBeOmitted addObject:fragment];
							  }
							  
							  // Skip others if requested
							  if ((extractorOptions & WordExtractorOptionOmitOthers) &&
								  (tag != NSLinguisticTagVerb) &&
								  (tag != NSLinguisticTagAdjective) &&
								  (tag != NSLinguisticTagAdverb) &&
								  (tag != NSLinguisticTagNumber) &&
								  (tag != NSLinguisticTagNoun)) {
								  
								  // Store for later removal
								  [fragmentsToBeOmitted addObject:fragment];
							  }
							  
							  // Check for 2-words combinations for kept fragments
							  if (fragments.count > 1) {
								  TextFragment *previousFragment= [fragments objectAtIndex:fragments.count -2];
								  
								  if ([fragment isContiguous:previousFragment]) {
									  if (extractorOptions & WordExtractorOptionKeepAllBigrams) {
										  
										  // Form a bigram with the previous fragment
										  TextFragment *combinedFragment= [fragment combineWithFragment:previousFragment];
										  [combinedFragments addObject:combinedFragment];
										  
										  if ((extractorOptions & WordExtractorOptionKeepAllTrigrams) && (fragments.count > 2))	{
											  TextFragment *previousPreviousFragment= [fragments objectAtIndex:fragments.count -3];
											  
											  if ([previousFragment isContiguous:previousPreviousFragment]) {
												  
												  // Form a trigram with the last two fragments
												  TextFragment *combinedFragment2= [combinedFragment combineWithFragment:previousPreviousFragment];
												  [combinedFragments addObject:combinedFragment2];
											  }
										  }
										  
									  } else if ((extractorOptions & WordExtractorOptionKeep2WordNames) &&
												 (tag == NSLinguisticTagNoun) &&
												 ([word isCapitalizedString] || [word isNameInitial]) &&
												 (previousFragment.linguisticTag == NSLinguisticTagNoun) &&
												 ([previousFragment.fragment isCapitalizedString] || [previousFragment.fragment isNameInitial])) {
										  
										  // Form a 2-words name with the previous fragment
										  TextFragment *combinedFragment= [fragment combineWithFragment:previousFragment];
										  [combinedFragments addObject:combinedFragment];
										  
										  if ((extractorOptions & WordExtractorOptionKeep3WordNames) && (fragments.count > 2)) {
											  TextFragment *previousPreviousFragment= [fragments objectAtIndex:fragments.count -3];
											  
											  if ([previousFragment isContiguous:previousPreviousFragment] &&
												  (previousPreviousFragment.linguisticTag == NSLinguisticTagNoun) &&
												  ([previousPreviousFragment.fragment isCapitalizedString] || [previousPreviousFragment.fragment isNameInitial])) {
												  
												  // Form a 3-words name with the last two fragments
												  TextFragment *combinedFragment2= [combinedFragment combineWithFragment:previousPreviousFragment];
												  [combinedFragments addObject:combinedFragment2];
											  }
										  }
										  
									  } else {
										  if ((extractorOptions & WordExtractorOptionKeepNounVerbCombos) &&
											  (tag == NSLinguisticTagVerb) &&
											  (previousFragment.linguisticTag == NSLinguisticTagNoun)) {
											  
											  // Form a noun-verb combo with the previous fragment
											  TextFragment *combinedFragment= [fragment combineWithFragment:previousFragment];
											  [combinedFragments addObject:combinedFragment];
										  }
										  
										  if ((extractorOptions & WordExtractorOptionKeepVerbAdjectiveCombos) &&
											  (tag == NSLinguisticTagAdjective) &&
											  (previousFragment.linguisticTag == NSLinguisticTagVerb)) {
											  
											  // Form a verb-adjective combo with the previous fragment
											  TextFragment *combinedFragment= [fragment combineWithFragment:previousFragment];
											  [combinedFragments addObject:combinedFragment];
										  }
										  
										  if ((extractorOptions & WordExtractorOptionKeepAdjectiveNounCombos) &&
											  (tag == NSLinguisticTagNoun) &&
											  (previousFragment.linguisticTag == NSLinguisticTagAdjective)) {
											  
											  // Form an adjective-noun combo with the previous fragment
											  TextFragment *combinedFragment= [fragment combineWithFragment:previousFragment];
											  [combinedFragments addObject:combinedFragment];
										  }
										  
										  if ((extractorOptions & WordExtractorOptionKeepAdverbNounCombos) &&
											  (tag == NSLinguisticTagNoun) &&
											  (previousFragment.linguisticTag == NSLinguisticTagAdverb)) {
											  
											  // Form an adverb-noun combo with the previous fragment
											  TextFragment *combinedFragment= [fragment combineWithFragment:previousFragment];
											  [combinedFragments addObject:combinedFragment];
										  }
										  
										  if ((extractorOptions & WordExtractorOptionKeepNounNounCombos) &&
											  (tag == NSLinguisticTagNoun) &&
											  (previousFragment.linguisticTag == NSLinguisticTagNoun)) {
											  
											  // Form a noun-noun combo with the previous fragment
											  TextFragment *combinedFragment= [fragment combineWithFragment:previousFragment];
											  [combinedFragments addObject:combinedFragment];
										  }
									  }
								  }
							  }
						  }];
		
		[fragments addObjectsFromArray:combinedFragments];
		combinedFragments= nil;
		
		// Remove fragments to be omitted
		for (TextFragment *fragment in fragmentsToBeOmitted)
			[fragments removeObject:fragment];

		fragmentsToBeOmitted= nil;
		
		// Sort fragments according to token index
		[fragments sortUsingComparator:^NSComparisonResult(id obj1, id obj2) {
			TextFragment *fragment1= (TextFragment *) obj1;
			TextFragment *fragment2= (TextFragment *) obj2;
			
			return (fragment1.tokenIndex < fragment2.tokenIndex) ? NSOrderedAscending :
					((fragment1.tokenIndex > fragment2.tokenIndex) ? NSOrderedDescending : NSOrderedSame);
		}];
		
		if (extractorOptions & WordExtractorOptionKeepEmoticons)
			[self insertEmoticonFragments:text fragments:fragments];

		// Return the tokens
		NSMutableArray *tokens= [[NSMutableArray alloc] initWithCapacity:fragments.count];
		
		for (TextFragment *fragment in fragments)
			[tokens addObject:fragment.fragment];
		
		return tokens;
	}
}

+ (NSArray *) extractTokensWithSimpleTokenizer:(NSString *)text language:(NSString *)languageCode extractorOptions:(WordExtractorOption)extractorOptions {
	@autoreleasepool {
	
		// Make sure full-stops and apostrophes are followed by spaces
		text= [text stringByReplacingOccurrencesOfString:@"." withString:@". "];
		text= [text stringByReplacingOccurrencesOfString:@"'" withString:@"' "];
		
		// Prepare containers and stopword list
		NSMutableArray *fragments= [NSMutableArray arrayWithCapacity:text.length / 5];
		NSMutableArray *combinedFragments= [NSMutableArray arrayWithCapacity:text.length / 10];
		NSSet *stopWords= [__stopWords objectForKey:languageCode];
		
		// Split text by spaces and new lines
		int tokenIndex= -1;
		NSRange range= NSMakeRange(0, text.length);
		do {
			NSRange sep= [text rangeOfCharacterFromSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]
											   options:0
												 range:range];
			
			if (sep.location == NSNotFound)
				sep.location= text.length;
			
			// Extract token
			NSRange tokenRange= NSMakeRange(range.location, sep.location - range.location);
			NSString *word= [text substringWithRange:tokenRange];
			
			// Update search range
			range.location= sep.location +1;
			range.length= text.length - range.location;
			
			// Clean up punctuation
			word= [word stringByTrimmingCharactersInSet:[NSCharacterSet punctuationCharacterSet]];
			if (word.length < 2)
				continue;
			
			tokenIndex++;
			
			// Skip stopwords if requested
			if ((extractorOptions & WordExtractorOptionOmitStopWords) &&
				[stopWords containsObject:[word lowercaseString]])
				continue;
			
			// Skip numbers if requested
			if ((extractorOptions & WordExtractorOptionOmitNumbers) &&
				[word intValue])
				continue;
			
			// Add the fragment
			TextFragment *fragment= [[TextFragment alloc] initWithFrament:word
																	range:tokenRange
															sentenceRange:NSMakeRange(0, text.length)
															   tokenIndex:tokenIndex
															linguisticTag:nil];
			
			[fragments addObject:fragment];
			
			// Check for 2-words combinations
			if ((extractorOptions & WordExtractorOptionKeepAllBigrams) && (fragments.count > 1)) {
				TextFragment *previousFragment= [fragments objectAtIndex:fragments.count -2];
				
				if ([fragment isContiguous:previousFragment]) {
					
					// Form a bigram with the previous fragment
					TextFragment *combinedFragment= [fragment combineWithFragment:previousFragment];
					[combinedFragments addObject:combinedFragment];
					
					if ((extractorOptions & WordExtractorOptionKeepAllTrigrams) && (fragments.count > 2))	{
						TextFragment *previousPreviousFragment= [fragments objectAtIndex:fragments.count -3];
						
						if ([previousFragment isContiguous:previousPreviousFragment]) {
							
							// Form a trigram with the last two fragments
							TextFragment *combinedFragment2= [combinedFragment combineWithFragment:previousPreviousFragment];
							[combinedFragments addObject:combinedFragment2];
						}
					}
				}
			}
			
		} while (range.location < text.length);
		
		[fragments addObjectsFromArray:combinedFragments];
		combinedFragments= nil;
		
		// Sort fragments according to token index
		[fragments sortUsingComparator:^NSComparisonResult(id obj1, id obj2) {
			TextFragment *fragment1= (TextFragment *) obj1;
			TextFragment *fragment2= (TextFragment *) obj2;
			
			return (fragment1.tokenIndex < fragment2.tokenIndex) ? NSOrderedAscending :
			((fragment1.tokenIndex > fragment2.tokenIndex) ? NSOrderedDescending : NSOrderedSame);
		}];
		
		if (extractorOptions & WordExtractorOptionKeepEmoticons)
			[BagOfWords insertEmoticonFragments:text fragments:fragments];
		
		// Return the tokens
		NSMutableArray *tokens= [[NSMutableArray alloc] initWithCapacity:fragments.count];
		
		for (TextFragment *fragment in fragments)
			[tokens addObject:fragment.fragment];
		
		return tokens;
	}
}


#pragma mark -
#pragma mark Extractor support

+ (void) insertEmoticonFragments:(NSString *)text fragments:(NSMutableArray *)fragments {
	NSMutableArray *matches= [NSMutableArray array];
	
	// Look for emoticons with a couple of regex
	NSRegularExpression *regex= [NSRegularExpression regularExpressionWithPattern:LEFT_TO_RIGHT_EMOTICON
																		  options:0
																			error:nil];
	
	[matches addObjectsFromArray:[regex matchesInString:text options:0 range:NSMakeRange(0, [text length])]];
	
	regex= [NSRegularExpression regularExpressionWithPattern:RIGHT_TO_LEFT_EMOTICON
													 options:0
													   error:nil];
	
	[matches addObjectsFromArray:[regex matchesInString:text options:0 range:NSMakeRange(0, [text length])]];
	
	regex= [NSRegularExpression regularExpressionWithPattern:EMOJI
													 options:0
													   error:nil];
	
	[matches addObjectsFromArray:[regex matchesInString:text options:0 range:NSMakeRange(0, [text length])]];
	
	// Now appropriately insert the emoticon in the right place using
	// binary search and checking the match location
	for (NSTextCheckingResult *match in matches) {
		NSUInteger pos= fragments.count / 2;
		NSUInteger span= pos / 2;
		
		while (span > 1) {
			TextFragment *fragment= [fragments objectAtIndex:pos];
			
			if (match.range.location < fragment.range.location) {
				pos -= span;
				span /= 2;
				
			} else if (match.range.location > fragment.range.location) {
				pos += span;
				span /= 2;
				
			} else
				break;
		}
		
		NSRange emoticonRange= NSMakeRange(match.range.location +1, match.range.length -1);
		TextFragment *emoticon= [[TextFragment alloc] initWithFrament:[text substringWithRange:emoticonRange]
																range:emoticonRange
														sentenceRange:NSMakeRange(0, [text length])
														   tokenIndex:0.0
														linguisticTag:nil];
		
		[fragments insertObject:emoticon atIndex:pos];
	}
}


#pragma mark -
#pragma mark Properties

@synthesize textID= _textID;
@synthesize tokens= _tokens;

@synthesize outputSize= _outputSize;
@synthesize outputBuffer= _outputBuffer;


@end

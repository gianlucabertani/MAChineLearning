//
//  MLBagOfWords.m
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

#import "MLBagOfWords.h"
#import "MLMutableWordDictionary.h"
#import "MLWordInfo.h"
#import "MLTextFragment.h"
#import "MLStopWords.h"
#import "NSString+WordUtils.h"
#import "MLBagOfWordsException.h"

#import "MLConstants.h"

#import <Accelerate/Accelerate.h>

#define LEFT_TO_RIGHT_EMOTICON             (@"\\s[:=;B]-?[)(|\\/\\\\\\]\\[DOoPp]")
#define RIGHT_TO_LEFT_EMOTICON             (@"\\s[)(|\\/\\\\\\]\\[DOo]-?[:=;]")
#define EMOJI                              (@"[😀😁😂😃😄😅😆😇😊😉👿😈☺️😋😌😍😑😐😏😎😒😓😔😕😙😘😗😖😚😛😜😝😡😠😟😞😢😣😤😥😩😨😧😦😪😫😬😭😱😰😯😮😲😳😴😵😶😷]")

#define GUESS_THRESHOLD_PERC              (10)


#pragma mark -
#pragma mark BagOfWords extension

@interface MLBagOfWords () {
	NSString *_documentID;
	NSArray *_words;

	NSUInteger _outputSize;
	MLReal *_outputBuffer;
	BOOL _localBuffer;
}


#pragma mark -
#pragma mark Extractor support

+ (NSString *) addSpaceInText:(NSString *)text afterCharactersInSet:(NSCharacterSet *)charSet;
+ (void) insertEmoticonFragments:(NSString *)text fragments:(NSMutableArray *)fragments;


#pragma mark -
#pragma mark Internals

- (void) prepareOutputBuffer:(MLReal *)outputBuffer;
- (void) fillOutputBuffer:(MLWordDictionary *)dictionary buildDictionary:(BOOL)buildDictionary featureNormalization:(MLFeatureNormalizationType)normalizationType;
- (void) normalizeOutputBuffer:(MLWordDictionary *)dictionary featureNormalization:(MLFeatureNormalizationType)normalizationType;


@end


#pragma mark -
#pragma mark BagOfWords statics

static NSDictionary *__stopWords= nil;


#pragma mark -
#pragma mark BagOfWords implementation

@implementation MLBagOfWords


#pragma mark -
#pragma mark Initialization

+ (MLBagOfWords *) bagOfWordsForTopicClassificationWithText:(NSString *)text documentID:(NSString *)documentID dictionary:(MLMutableWordDictionary *)dictionary language:(NSString *)languageCode featureNormalization:(MLFeatureNormalizationType)normalizationType {
	return [MLBagOfWords bagOfWordsWithText:text
									documentID:documentID
								dictionary:dictionary
							buildDictionary:YES
								   language:languageCode
							  wordExtractor:MLWordExtractorTypeLinguisticTagger
						   extractorOptions:MLWordExtractorOptionOmitStopWords | MLWordExtractorOptionOmitVerbs | MLWordExtractorOptionOmitAdjectives | MLWordExtractorOptionOmitAdverbs | MLWordExtractorOptionOmitNouns | MLWordExtractorOptionOmitOthers | MLWordExtractorOptionKeepAdjectiveNounCombos | MLWordExtractorOptionKeepAdverbNounCombos | MLWordExtractorOptionKeepNounNounCombos | MLWordExtractorOptionKeep2WordNames | MLWordExtractorOptionKeep3WordNames
					   featureNormalization:normalizationType
							   outputBuffer:nil];
}

+ (MLBagOfWords *) bagOfWordsForSentimentAnalysisWithText:(NSString *)text documentID:(NSString *)documentID dictionary:(MLMutableWordDictionary *)dictionary language:(NSString *)languageCode featureNormalization:(MLFeatureNormalizationType)normalizationType {
	return [MLBagOfWords bagOfWordsWithText:text
									documentID:documentID
								dictionary:dictionary
							buildDictionary:YES
								   language:languageCode
							  wordExtractor:MLWordExtractorTypeSimpleTokenizer
						   extractorOptions:MLWordExtractorOptionOmitStopWords | MLWordExtractorOptionKeepEmoticons | MLWordExtractorOptionKeepAllBigrams
					   featureNormalization:normalizationType
							   outputBuffer:nil];
}

+ (MLBagOfWords *) bagOfWordsWithText:(NSString *)text documentID:(NSString *)documentID dictionary:(MLWordDictionary *)dictionary buildDictionary:(BOOL)buildDictionary language:(NSString *)languageCode wordExtractor:(MLWordExtractorType)extractorType extractorOptions:(MLWordExtractorOption)extractorOptions featureNormalization:(MLFeatureNormalizationType)normalizationType outputBuffer:(MLReal *)outputBuffer {
	MLBagOfWords *bagOfWords= [[MLBagOfWords alloc] initWithText:text
														  documentID:documentID
													  dictionary:dictionary
												 buildDictionary:buildDictionary
														language:languageCode
												   wordExtractor:extractorType
												extractorOptions:extractorOptions
											featureNormalization:normalizationType
													outputBuffer:outputBuffer];
	
	return bagOfWords;
}

+ (MLBagOfWords *) bagOfWordsWithWords:(NSArray *)words documentID:(NSString *)documentID dictionary:(MLWordDictionary *)dictionary buildDictionary:(BOOL)buildDictionary featureNormalization:(MLFeatureNormalizationType)normalizationType outputBuffer:(MLReal *)outputBuffer {
	MLBagOfWords *bagOfWords= [[MLBagOfWords alloc] initWithWords:words
														   documentID:documentID
													   dictionary:dictionary
												  buildDictionary:buildDictionary
											 featureNormalization:normalizationType
													 outputBuffer:outputBuffer];
	
	return bagOfWords;
}

- (instancetype) initWithText:(NSString *)text documentID:(NSString *)documentID dictionary:(MLWordDictionary *)dictionary buildDictionary:(BOOL)buildDictionary language:(NSString *)languageCode wordExtractor:(MLWordExtractorType)extractorType extractorOptions:(MLWordExtractorOption)extractorOptions featureNormalization:(MLFeatureNormalizationType)normalizationType outputBuffer:(MLReal *)outputBuffer {
	if ((self = [super init])) {
		
		// Fill stop words if not filled already
		if (!__stopWords)
			__stopWords= ML_STOP_WORDS;
		
		// Checks
		if (!dictionary)
			@throw [MLBagOfWordsException bagOfWordsExceptionWithReason:@"Missing dictionary"
															 userInfo:nil];
		
		if (buildDictionary && (![dictionary isKindOfClass:[MLMutableWordDictionary class]]))
			@throw [MLBagOfWordsException bagOfWordsExceptionWithReason:@"Supplied dictionary is not mutable"
															   userInfo:nil];
		
		if (((extractorOptions & MLWordExtractorOptionOmitVerbs) |
			 (extractorOptions & MLWordExtractorOptionOmitAdjectives) |
			 (extractorOptions & MLWordExtractorOptionOmitAdverbs) |
			 (extractorOptions & MLWordExtractorOptionOmitNouns) |
			 (extractorOptions & MLWordExtractorOptionOmitNames) |
			 (extractorOptions & MLWordExtractorOptionOmitOthers) |
			 (extractorOptions & MLWordExtractorOptionKeepAdjectiveNounCombos) |
			 (extractorOptions & MLWordExtractorOptionKeepAdverbNounCombos) |
			 (extractorOptions & MLWordExtractorOptionKeepNounNounCombos) |
			 (extractorOptions & MLWordExtractorOptionKeepNounVerbCombos) |
			 (extractorOptions & MLWordExtractorOptionKeepVerbAdjectiveCombos) |
			 (extractorOptions & MLWordExtractorOptionKeep2WordNames) |
			 (extractorOptions & MLWordExtractorOptionKeep3WordNames)) &&
			(extractorType != MLWordExtractorTypeLinguisticTagger))
			@throw [MLBagOfWordsException bagOfWordsExceptionWithReason:@"Options on verbs, adjectives, adverbs, nouns and names require the linguistic tagger"
															 userInfo:nil];
		
		switch (normalizationType) {
			case MLFeatureNormalizationTypeL2TFiDF:
				if (buildDictionary)
					@throw [MLBagOfWordsException bagOfWordsExceptionWithReason:@"TF-iDF normalization requires a pre-built dictionary"
																	 userInfo:nil];
				
			default:
				break;
		}
		
		// Initialization
		_documentID= [documentID copy];
		
		// Run the appropriate extractor
		switch (extractorType) {
			case MLWordExtractorTypeSimpleTokenizer:
				_words= [MLBagOfWords extractWordsWithSimpleTokenizerFromText:text withLanguage:languageCode extractorOptions:extractorOptions];
				break;
				
			case MLWordExtractorTypeLinguisticTagger:
				_words= [MLBagOfWords extractWordsWithLinguisticTaggerFromText:text withLanguage:languageCode extractorOptions:extractorOptions];
				break;
		}
		
		_outputSize= (buildDictionary ? [(MLMutableWordDictionary *) dictionary maxSize] : [dictionary size]);
		
		// Set up the output buffer
		[self prepareOutputBuffer:outputBuffer];
		
		// Build dictionary and the output buffer
		[self fillOutputBuffer:dictionary buildDictionary:buildDictionary featureNormalization:normalizationType];

		// Apply vector-wide normalization
		[self normalizeOutputBuffer:dictionary featureNormalization:normalizationType];
	}
	
	return self;
}

- (instancetype) initWithWords:(NSArray *)words documentID:(NSString *)documentID dictionary:(MLWordDictionary *)dictionary buildDictionary:(BOOL)buildDictionary featureNormalization:(MLFeatureNormalizationType)normalizationType outputBuffer:(MLReal *)outputBuffer {
	if ((self = [super init])) {
		
		// Checks
		if (!dictionary)
			@throw [MLBagOfWordsException bagOfWordsExceptionWithReason:@"Missing dictionary"
															 userInfo:nil];
		
		if (buildDictionary && (![dictionary isKindOfClass:[MLMutableWordDictionary class]]))
			@throw [MLBagOfWordsException bagOfWordsExceptionWithReason:@"Supplied dictionary is not mutable"
															   userInfo:nil];
		
		switch (normalizationType) {
			case MLFeatureNormalizationTypeL2TFiDF:
				if (buildDictionary)
					@throw [MLBagOfWordsException bagOfWordsExceptionWithReason:@"TF-iDF normalization requires a pre-built dictionary"
																	 userInfo:nil];
				
			default:
				break;
		}
	
		// Initialization
		_words= [NSArray arrayWithArray:words];
		_documentID= [documentID copy];

		_outputSize= (buildDictionary ? [(MLMutableWordDictionary *) dictionary maxSize] : [dictionary size]);

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

- (void) prepareOutputBuffer:(MLReal *)outputBuffer {
	if (outputBuffer) {
		_outputBuffer= outputBuffer;
		_localBuffer= NO;
		
	} else {
		int err= posix_memalign((void **) &_outputBuffer,
								BUFFER_MEMORY_ALIGNMENT,
								sizeof(MLReal) * _outputSize);
		if (err)
			@throw [MLBagOfWordsException bagOfWordsExceptionWithReason:@"Error while allocating buffer"
															 userInfo:@{@"buffer": @"outputBuffer",
																		@"error": [NSNumber numberWithInt:err]}];
		
		_localBuffer= YES;
	}
	
	// Clear the output buffer
	ML_VDSP_VCLR(_outputBuffer, 1, _outputSize);
}

- (void) fillOutputBuffer:(MLWordDictionary *)dictionary buildDictionary:(BOOL)buildDictionary featureNormalization:(MLFeatureNormalizationType)normalizationType {
	
	// Build dictionary and the output buffer
	for (NSString *word in _words) {
		if (buildDictionary)
			[(MLMutableWordDictionary *) dictionary countOccurrenceForWord:word documentID:_documentID];

		MLWordInfo *wordInfo= [dictionary infoForWord:word];
		
		if (wordInfo) {
			switch (normalizationType) {
				case MLFeatureNormalizationTypeNone:
				case MLFeatureNormalizationTypeL2:
				case MLFeatureNormalizationTypeL2TFiDF:
					_outputBuffer[wordInfo.position] += 1.0;
					break;
					
				case MLFeatureNormalizationTypeBoolean:
					_outputBuffer[wordInfo.position]= 1.0;
					break;
			}
		}
	}
}

- (void) normalizeOutputBuffer:(MLWordDictionary *)dictionary featureNormalization:(MLFeatureNormalizationType)normalizationType {
	
	// Apply vector-wide normalization
	switch (normalizationType) {
		case MLFeatureNormalizationTypeL2TFiDF: {
			
			// Multiply by IDF weights (the dictionary computes
			// them on demand, then keeps them cached)
			ML_VDSP_VMUL(_outputBuffer, 1, dictionary.idfWeights, 1, _outputBuffer, 1, _outputSize);
			
			// NOTE: No "break" intended here
		}
			
		case MLFeatureNormalizationTypeL2: {
			MLReal normL2= 0.0;
			ML_VDSP_SVESQ(_outputBuffer, 1, &normL2, _outputSize);
			normL2= ML_SQRT(normL2);

			ML_VDSP_VSDIV(_outputBuffer, 1, &normL2, _outputBuffer, 1, _outputSize);
			break;
		}
			
		case MLFeatureNormalizationTypeBoolean:
		case MLFeatureNormalizationTypeNone:
			break;
	}
}


#pragma mark -
#pragma mark Dictionary building

+ (void) buildDictionaryWithText:(NSString *)text documentID:(NSString *)documentID dictionary:(MLMutableWordDictionary *)dictionary language:(NSString *)languageCode wordExtractor:(MLWordExtractorType)extractorType extractorOptions:(MLWordExtractorOption)extractorOptions {
	NSArray *words= nil;
	
	// Run the appropriate word extractor
	switch (extractorType) {
		case MLWordExtractorTypeSimpleTokenizer:
			words= [MLBagOfWords extractWordsWithSimpleTokenizerFromText:text withLanguage:languageCode extractorOptions:extractorOptions];
			break;
			
		case MLWordExtractorTypeLinguisticTagger:
			words= [MLBagOfWords extractWordsWithLinguisticTaggerFromText:text withLanguage:languageCode extractorOptions:extractorOptions];
			break;
	}
	
	for (NSString *word in words)
		[dictionary countOccurrenceForWord:word documentID:documentID];
}


#pragma mark -
#pragma mark Languages code guessing

+ (NSString *) guessLanguageCodeWithLinguisticTaggerForText:(NSString *)text {
	@autoreleasepool {
	
		// Init the linguistic tagger
		NSLinguisticTagger *tagger= [[NSLinguisticTagger alloc] initWithTagSchemes:@[NSLinguisticTagSchemeLanguage] options:0];
		[tagger setString:text];
	
		// Get the language using the tagger
		NSString *language= [tagger tagAtIndex:0 scheme:NSLinguisticTagSchemeLanguage tokenRange:NULL sentenceRange:NULL];
		return language;
	}
}

+ (NSString *) guessLanguageCodeWithStopWordsForText:(NSString *)text {
	@autoreleasepool {
	
		// Fill stop words if not filled already
		if (!__stopWords)
			__stopWords= ML_STOP_WORDS;
		
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

+ (NSArray *) extractWordsWithLinguisticTaggerFromText:(NSString *)text withLanguage:(NSString *)languageCode extractorOptions:(MLWordExtractorOption)extractorOptions {

	// Checks
	if ((extractorOptions & MLWordExtractorOptionOmitStopWords) &&
		(!languageCode))
		@throw [MLBagOfWordsException bagOfWordsExceptionWithReason:@"Missing language code (language is needed to skip stopwords)"
														   userInfo:nil];
	
	@autoreleasepool {
		
		// Fill stop words if not filled already
		if (!__stopWords)
			__stopWords= ML_STOP_WORDS;
		
		// Keep a pointer to the original text for later search of emoticons
		NSString *originalText= text;
	
		// Ensure punctuation is followed by a space
		text= [MLBagOfWords addSpaceInText:text afterCharactersInSet:[NSCharacterSet punctuationCharacterSet]];
		
		// Prepare containers and stopwords list
		NSMutableArray *fragments= [NSMutableArray arrayWithCapacity:text.length / 5];
		NSMutableArray *combinedFragments= [NSMutableArray arrayWithCapacity:text.length / 10];
		NSMutableArray *fragmentsToBeOmitted= [NSMutableArray arrayWithCapacity:text.length / 10];
		NSSet *stopWords= (languageCode ? [__stopWords objectForKey:languageCode] : nil);
		
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
							  if ((extractorOptions & MLWordExtractorOptionOmitStopWords) &&
								  [stopWords containsObject:[word lowercaseString]])
								  return;

							  // Create the fragment
							  MLTextFragment *fragment= [[MLTextFragment alloc] initWithFrament:word
																					  range:tokenRange
																			  sentenceRange:sentenceRange
																				 tokenIndex:tokenIndex
																			  linguisticTag:tag];
							  
							  [fragments addObject:fragment];
							  
							  // Skip verbs if requested
							  if ((extractorOptions & MLWordExtractorOptionOmitVerbs) &&
								  (tag == NSLinguisticTagVerb)) {

								  // Store for later removal
								  [fragmentsToBeOmitted addObject:fragment];
							  }
							  
							  // Skip adjectives if requested
							  if ((extractorOptions & MLWordExtractorOptionOmitAdjectives) &&
								  (tag == NSLinguisticTagAdjective)) {
								  
								  // Store for later removal
								  [fragmentsToBeOmitted addObject:fragment];
							  }
							  
							  // Skip adverbs if requested
							  if ((extractorOptions & MLWordExtractorOptionOmitAdverbs) &&
								  (tag == NSLinguisticTagAdverb)) {
								  
								  // Store for later removal
								  [fragmentsToBeOmitted addObject:fragment];
							  }
							  
							  // Skip nouns if requested
							  if ((extractorOptions & MLWordExtractorOptionOmitNouns) &&
								  (tag == NSLinguisticTagNoun) &&
								  (!([word isCapitalizedString] || [word isNameInitial] || [word isAcronym]))) {
								  
								  // Store for later removal
								  [fragmentsToBeOmitted addObject:fragment];
							  }

							  // Skip proper names if requested
							  if ((extractorOptions & MLWordExtractorOptionOmitNames) &&
								  (tag == NSLinguisticTagNoun) &&
								  ([word isCapitalizedString] || [word isNameInitial] || [word isAcronym])) {
								  
								  // Store for later removal
								  [fragmentsToBeOmitted addObject:fragment];
							  }
							  
							  // Skip numbers if requested
							  if ((extractorOptions & MLWordExtractorOptionOmitNumbers) &&
								  (tag == NSLinguisticTagNumber)) {
								  
								  // Store for later removal
								  [fragmentsToBeOmitted addObject:fragment];
							  }
							  
							  // Skip others if requested
							  if ((extractorOptions & MLWordExtractorOptionOmitOthers) &&
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
								  MLTextFragment *previousFragment= [fragments objectAtIndex:fragments.count -2];
								  
								  if ([fragment isContiguous:previousFragment]) {
									  if (extractorOptions & MLWordExtractorOptionKeepAllBigrams) {
										  
										  // Form a bigram with the previous fragment
										  MLTextFragment *combinedFragment= [fragment combineWithFragment:previousFragment];
										  [combinedFragments addObject:combinedFragment];
										  
										  if ((extractorOptions & MLWordExtractorOptionKeepAllTrigrams) && (fragments.count > 2))	{
											  MLTextFragment *previousPreviousFragment= [fragments objectAtIndex:fragments.count -3];
											  
											  if ([previousFragment isContiguous:previousPreviousFragment]) {
												  
												  // Form a trigram with the last two fragments
												  MLTextFragment *combinedFragment2= [combinedFragment combineWithFragment:previousPreviousFragment];
												  [combinedFragments addObject:combinedFragment2];
											  }
										  }
										  
									  } else if ((extractorOptions & MLWordExtractorOptionKeep2WordNames) &&
												 (tag == NSLinguisticTagNoun) &&
												 ([word isCapitalizedString] || [word isNameInitial]) &&
												 (previousFragment.linguisticTag == NSLinguisticTagNoun) &&
												 ([previousFragment.fragment isCapitalizedString] || [previousFragment.fragment isNameInitial])) {
										  
										  // Form a 2-words name with the previous fragment
										  MLTextFragment *combinedFragment= [fragment combineWithFragment:previousFragment];
										  [combinedFragments addObject:combinedFragment];
										  
										  if ((extractorOptions & MLWordExtractorOptionKeep3WordNames) && (fragments.count > 2)) {
											  MLTextFragment *previousPreviousFragment= [fragments objectAtIndex:fragments.count -3];
											  
											  if ([previousFragment isContiguous:previousPreviousFragment] &&
												  (previousPreviousFragment.linguisticTag == NSLinguisticTagNoun) &&
												  ([previousPreviousFragment.fragment isCapitalizedString] || [previousPreviousFragment.fragment isNameInitial])) {
												  
												  // Form a 3-words name with the last two fragments
												  MLTextFragment *combinedFragment2= [combinedFragment combineWithFragment:previousPreviousFragment];
												  [combinedFragments addObject:combinedFragment2];
											  }
										  }
										  
									  } else {
										  if ((extractorOptions & MLWordExtractorOptionKeepNounVerbCombos) &&
											  (tag == NSLinguisticTagVerb) &&
											  (previousFragment.linguisticTag == NSLinguisticTagNoun)) {
											  
											  // Form a noun-verb combo with the previous fragment
											  MLTextFragment *combinedFragment= [fragment combineWithFragment:previousFragment];
											  [combinedFragments addObject:combinedFragment];
										  }
										  
										  if ((extractorOptions & MLWordExtractorOptionKeepVerbAdjectiveCombos) &&
											  (tag == NSLinguisticTagAdjective) &&
											  (previousFragment.linguisticTag == NSLinguisticTagVerb)) {
											  
											  // Form a verb-adjective combo with the previous fragment
											  MLTextFragment *combinedFragment= [fragment combineWithFragment:previousFragment];
											  [combinedFragments addObject:combinedFragment];
										  }
										  
										  if ((extractorOptions & MLWordExtractorOptionKeepAdjectiveNounCombos) &&
											  (tag == NSLinguisticTagNoun) &&
											  (previousFragment.linguisticTag == NSLinguisticTagAdjective)) {
											  
											  // Form an adjective-noun combo with the previous fragment
											  MLTextFragment *combinedFragment= [fragment combineWithFragment:previousFragment];
											  [combinedFragments addObject:combinedFragment];
										  }
										  
										  if ((extractorOptions & MLWordExtractorOptionKeepAdverbNounCombos) &&
											  (tag == NSLinguisticTagNoun) &&
											  (previousFragment.linguisticTag == NSLinguisticTagAdverb)) {
											  
											  // Form an adverb-noun combo with the previous fragment
											  MLTextFragment *combinedFragment= [fragment combineWithFragment:previousFragment];
											  [combinedFragments addObject:combinedFragment];
										  }
										  
										  if ((extractorOptions & MLWordExtractorOptionKeepNounNounCombos) &&
											  (tag == NSLinguisticTagNoun) &&
											  (previousFragment.linguisticTag == NSLinguisticTagNoun)) {
											  
											  // Form a noun-noun combo with the previous fragment
											  MLTextFragment *combinedFragment= [fragment combineWithFragment:previousFragment];
											  [combinedFragments addObject:combinedFragment];
										  }
									  }
								  }
							  }
						  }];
		
		[fragments addObjectsFromArray:combinedFragments];
		combinedFragments= nil;
		
		// Remove fragments to be omitted
		for (MLTextFragment *fragment in fragmentsToBeOmitted)
			[fragments removeObject:fragment];

		fragmentsToBeOmitted= nil;
		
		// Sort fragments according to token index
		[fragments sortUsingComparator:^NSComparisonResult(id obj1, id obj2) {
			MLTextFragment *fragment1= (MLTextFragment *) obj1;
			MLTextFragment *fragment2= (MLTextFragment *) obj2;
			
			return (fragment1.tokenIndex < fragment2.tokenIndex) ? NSOrderedAscending :
					((fragment1.tokenIndex > fragment2.tokenIndex) ? NSOrderedDescending : NSOrderedSame);
		}];
		
		if (extractorOptions & MLWordExtractorOptionKeepEmoticons)
			[self insertEmoticonFragments:originalText fragments:fragments];

		// Return the tokens
		NSMutableArray *words= [[NSMutableArray alloc] initWithCapacity:fragments.count];
		
		for (MLTextFragment *fragment in fragments)
			[words addObject:fragment.fragment];
		
		return words;
	}
}

+ (NSArray *) extractWordsWithSimpleTokenizerFromText:(NSString *)text withLanguage:(NSString *)languageCode extractorOptions:(MLWordExtractorOption)extractorOptions {
	
	// Checks
	if ((extractorOptions & MLWordExtractorOptionOmitStopWords) &&
		(!languageCode))
		@throw [MLBagOfWordsException bagOfWordsExceptionWithReason:@"Missing language code (language is needed to skip stopwords)"
														   userInfo:nil];

	@autoreleasepool {
		
		// Fill stop words if not filled already
		if (!__stopWords)
			__stopWords= ML_STOP_WORDS;
	
		// Keep a pointer to the original text for later search of emoticons
		NSString *originalText= text;

		// Ensure punctuation is followed by a space
		text= [MLBagOfWords addSpaceInText:text afterCharactersInSet:[NSCharacterSet punctuationCharacterSet]];
		
		// Prepare containers and stopword list
		NSMutableArray *fragments= [NSMutableArray arrayWithCapacity:text.length / 5];
		NSMutableArray *combinedFragments= [NSMutableArray arrayWithCapacity:text.length / 10];
		NSSet *stopWords= (languageCode ? [__stopWords objectForKey:languageCode] : nil);

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
			if ((extractorOptions & MLWordExtractorOptionOmitStopWords) &&
				[stopWords containsObject:[word lowercaseString]])
				continue;
			
			// Skip numbers if requested
			if ((extractorOptions & MLWordExtractorOptionOmitNumbers) &&
				[word intValue])
				continue;
			
			// Add the fragment
			MLTextFragment *fragment= [[MLTextFragment alloc] initWithFrament:word
																	range:tokenRange
															sentenceRange:NSMakeRange(0, text.length)
															   tokenIndex:tokenIndex
															linguisticTag:nil];
			
			[fragments addObject:fragment];
			
			// Check for 2-words combinations
			if ((extractorOptions & MLWordExtractorOptionKeepAllBigrams) && (fragments.count > 1)) {
				MLTextFragment *previousFragment= [fragments objectAtIndex:fragments.count -2];
				
				if ([fragment isContiguous:previousFragment]) {
					
					// Form a bigram with the previous fragment
					MLTextFragment *combinedFragment= [fragment combineWithFragment:previousFragment];
					[combinedFragments addObject:combinedFragment];
					
					if ((extractorOptions & MLWordExtractorOptionKeepAllTrigrams) && (fragments.count > 2))	{
						MLTextFragment *previousPreviousFragment= [fragments objectAtIndex:fragments.count -3];
						
						if ([previousFragment isContiguous:previousPreviousFragment]) {
							
							// Form a trigram with the last two fragments
							MLTextFragment *combinedFragment2= [combinedFragment combineWithFragment:previousPreviousFragment];
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
			MLTextFragment *fragment1= (MLTextFragment *) obj1;
			MLTextFragment *fragment2= (MLTextFragment *) obj2;
			
			return (fragment1.tokenIndex < fragment2.tokenIndex) ? NSOrderedAscending :
			((fragment1.tokenIndex > fragment2.tokenIndex) ? NSOrderedDescending : NSOrderedSame);
		}];
		
		if (extractorOptions & MLWordExtractorOptionKeepEmoticons)
			[MLBagOfWords insertEmoticonFragments:originalText fragments:fragments];
		
		// Return the tokens
		NSMutableArray *words= [[NSMutableArray alloc] initWithCapacity:fragments.count];
		
		for (MLTextFragment *fragment in fragments)
			[words addObject:fragment.fragment];
		
		return words;
	}
}


#pragma mark -
#pragma mark Extractor support

+ (NSString *) addSpaceInText:(NSString *)text afterCharactersInSet:(NSCharacterSet *)charSet {
	NSMutableString *expandedText= [[NSMutableString alloc] initWithCapacity:text.length * 1.2];
	
	NSRange range= NSMakeRange(0, text.length);
	do {
		NSRange charPos= [text rangeOfCharacterFromSet:charSet options:0 range:range];
		if (charPos.location == NSNotFound) {
			[expandedText appendString:[text substringWithRange:range]];
			break;
		}
		
		[expandedText appendString:[text substringWithRange:NSMakeRange(range.location, charPos.location +1 - range.location)]];
		[expandedText appendString:@" "];
		
		range= NSMakeRange(charPos.location +1, text.length - (charPos.location +1));
		
	} while (YES);
	
	return expandedText;
}

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
			MLTextFragment *fragment= [fragments objectAtIndex:pos];
			
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
		MLTextFragment *emoticon= [[MLTextFragment alloc] initWithFrament:[text substringWithRange:emoticonRange]
																range:emoticonRange
														sentenceRange:NSMakeRange(0, [text length])
														   tokenIndex:0.0
														linguisticTag:nil];
		
		[fragments insertObject:emoticon atIndex:pos];
	}
}


#pragma mark -
#pragma mark Properties

@synthesize documentID= _documentID;
@synthesize words= _words;

@synthesize outputSize= _outputSize;
@synthesize outputBuffer= _outputBuffer;


@end

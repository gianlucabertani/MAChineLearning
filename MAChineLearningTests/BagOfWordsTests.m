//
//  BagOfWordsTests.m
//  MAChineLearning
//
//  Created by Gianluca Bertani on 26/04/15.
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

#import <Foundation/Foundation.h>
#import <XCTest/XCTest.h>
#import <MAChineLearning/MAChineLearning.h>

// Source: Wikiquote
#define WOODY_ALLEN                        (@"If you're not failing every now and again, it's a sign you're not doing anything very innovative.")
#define DANTE_ALIGHIERI                    (@"Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura che la diritta via era smarrita. ")
#define MIGUEL_DE_CERVANTES                (@"La libertad es uno de los más preciosos dones que a los hombres dieron los cielos; " \
											@"con ella no pueden igualarse los tesoros que encierran la tierra y el mar: por la libertad se puede y debe aventurar la vida.")
#define CHARLES_BAUDELAIRE                 (@"Toutes les beautés contiennent, comme tous les phénomènes possibles, quelque chose d'éternel et quelque chose de transitoire — " \
											@"d'absolu et de particulier.")
#define ALBERT_EINSTEIN                    (@"Ich glaube an spinozas gott der sich in der harmonie des seienden offenbart stop nicht an einen gott der sich mit schicksalen " \
											@"und handlungen der menschen abgibt.")

// Source: Kaggle data for "Bag of Words meets Bag of Popcorns" test
#define MOVIE_REVIEW                       (@"I think the majority of the people seem not the get the right idea about the movie, at least that's my opinion. " \
											@"I am not sure it's a movie about drug abuse; rather it's a movie about the way of thinking of those genius brothers, " \
											@"drugs are side effects, something marginal. Again, it's not a commercial movie that you see every day and if the author " \
											@"wanted that, he definitely failed, as most people think it's one of the many drug related movies. I, however, think " \
											@"something else is the case. As in many movies portraying different cultures, audience usually fully understands movies " \
											@"portraying their own culture, i.e. something they've grown up with and are quite familiar with. This movie is to show what " \
											@"those \"genius\" people very often think and what problems they face. The reason why they act like this is because they are " \
											@"bored out of their minds :) They have to meet people who do mediocre things and accept those things as if they are launching " \
											@"space shuttles on daily basis. They start a fairly hard job and excel in no time. They feel like- I went to work, did nothing, " \
											@"still did twice as better as the guys around me when they were all over their projects, what should I do now with my free time. " \
											@"And what's even more boring? When you can start predicting behavior not because you're psychologist, but instead because you have " \
											@"seen this pattern in the past. So, for them, from one side it's a non challenging job, which is also fairly boring sometimes, " \
											@"and from another they start to figure out people's behavior. It's a recipe for big big boredom. And the dumbest things are usually " \
											@"done to get out of this state. They guy earlier who mentioned that their biggest problem is that they are trying to figure out " \
											@"life in terms of logic (math describes logic), while life is not really a logical thing, is actually absolutely right.")

// Source: Wikipedia page for Natural Language Processing
#define ARTICLE_EXTRACT                    (@"The history of NLP generally starts in the 1950s, although work can be found from earlier periods. In 1950, Alan Turing published " \
											@"an article titled \"Computing Machinery and Intelligence\" which proposed what is now called the Turing test as a criterion " \
											@"of intelligence. The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian sentences " \
											@"into English. The authors claimed that within three or five years, machine translation would be a solved problem. However, " \
											@"real progress was much slower, and after the ALPAC report in 1966, which found that ten-year-long research had failed to fulfill " \
											@"the expectations, funding for machine translation was dramatically reduced. Little further research in machine translation was " \
											@"conducted until the late 1980s, when the first statistical machine translation systems were developed. Some notably successful " \
											@"NLP systems developed in the 1960s were SHRDLU, a natural language system working in restricted \"blocks worlds\" with restricted " \
											@"vocabularies, and ELIZA, a simulation of a Rogerian psychotherapist, written by Joseph Weizenbaum between 1964 to 1966. Using " \
											@"almost no information about human thought or emotion, ELIZA sometimes provided a startlingly human-like interaction. When the " \
											@"\"patient\" exceeded the very small knowledge base, ELIZA might provide a generic response, for example, responding to " \
											@"\"My head hurts\" with \"Why do you say your head hurts?\".")


#pragma mark -
#pragma mark BagOfWordsTests declaration

@interface BagOfWordsTests : XCTestCase
@end


#pragma mark -
#pragma mark BagOfWordsTests implementation

@implementation BagOfWordsTests


#pragma mark -
#pragma mark Setup and tear down

- (void) setUp {
    [super setUp];
}

- (void) tearDown {
    [super tearDown];
}


#pragma mark -
#pragma mark Tests

- (void) testGuessLanguageWithLinguisticTagger {
	@try {
		NSString *lang1= [BagOfWords guessLanguageCodeWithLinguisticTagger:WOODY_ALLEN];
		XCTAssertEqualObjects(lang1, @"en");
		
		NSString *lang2= [BagOfWords guessLanguageCodeWithLinguisticTagger:DANTE_ALIGHIERI];
		XCTAssertEqualObjects(lang2, @"it");
		
		NSString *lang3= [BagOfWords guessLanguageCodeWithLinguisticTagger:MIGUEL_DE_CERVANTES];
		XCTAssertEqualObjects(lang3, @"es");
		
		NSString *lang4= [BagOfWords guessLanguageCodeWithLinguisticTagger:CHARLES_BAUDELAIRE];
		XCTAssertEqualObjects(lang4, @"fr");
		
		NSString *lang5= [BagOfWords guessLanguageCodeWithLinguisticTagger:ALBERT_EINSTEIN];
		XCTAssertEqualObjects(lang5, @"de");

	} @catch (NSException *e) {
		XCTFail(@"Exception caught while testing: %@, reason: '%@', user info: %@", e.name, e.reason, e.userInfo);
	}
}

- (void) testGuessLanguageWithStopWords {
	@try {
		NSString *lang1= [BagOfWords guessLanguageCodeWithStopWords:WOODY_ALLEN];
		XCTAssertEqualObjects(lang1, @"en");
		
		NSString *lang2= [BagOfWords guessLanguageCodeWithStopWords:DANTE_ALIGHIERI];
		XCTAssertEqualObjects(lang2, @"it");
		
		NSString *lang3= [BagOfWords guessLanguageCodeWithStopWords:MIGUEL_DE_CERVANTES];
		XCTAssertEqualObjects(lang3, @"es");
		
		NSString *lang4= [BagOfWords guessLanguageCodeWithStopWords:CHARLES_BAUDELAIRE];
		XCTAssertEqualObjects(lang4, @"fr");
		
		NSString *lang5= [BagOfWords guessLanguageCodeWithStopWords:ALBERT_EINSTEIN];
		XCTAssertEqualObjects(lang5, @"de");
		
	} @catch (NSException *e) {
		XCTFail(@"Exception caught while testing: %@, reason: '%@', user info: %@", e.name, e.reason, e.userInfo);
	}
}

- (void) testBagOfWordsForClassification {
	@try {
		NSMutableDictionary *dictionary= [NSMutableDictionary dictionary];
		
		BagOfWords *bag= [BagOfWords bagOfWordsForClassificationWithText:ARTICLE_EXTRACT
																  textID:@"article1"
															  dictionary:dictionary
														  dictionarySize:100
																language:@"en"
													featureNormalization:FeatureNormalizationTypeL2];

		XCTAssertTrue([bag.tokens containsObject:@"Alan Turing"]);

		// !! TO DO: to be completed
	
	} @catch (NSException *e) {
		XCTFail(@"Exception caught while testing: %@, reason: '%@', user info: %@", e.name, e.reason, e.userInfo);
	}
}


- (void) testBagOfWordsForSentimentAnalysis {
	@try {
		NSMutableDictionary *dictionary= [NSMutableDictionary dictionary];
		
		BagOfWords *bag= [BagOfWords bagOfWordsForSentimentAnalysisWithText:MOVIE_REVIEW
																	 textID:@"review1"
																 dictionary:dictionary
															 dictionarySize:200
																   language:@"en"
													   featureNormalization:FeatureNormalizationTypeL2];
		
		XCTAssertTrue([bag.tokens containsObject:@":)"]);
		
		// !! TO DO: to be completed

	} @catch (NSException *e) {
		XCTFail(@"Exception caught while testing: %@, reason: '%@', user info: %@", e.name, e.reason, e.userInfo);
	}
}

@end

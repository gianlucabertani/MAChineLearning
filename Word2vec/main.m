//
//  main.m
//  Word2vec
//
//  Created by Gianluca Bertani on 04/05/15.
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
#import <MAChineLearning/MAChineLearning.h>

#define DICTIONARY_SIZE                (10000)
#define VECTOR_SIZE                      (100)
#define SKIP_GRAM_WINDOW                   (5)
#define DOWNSAMPLING_FREQUENCY_LIMIT       (0.001)
#define LEARNING_RATE                      (0.025)

#define RETVAL_MISSING_ARGUMENT            (9)
#define RETVAL_BUFFER_ALLOCATION_ERROR    (17)
#define RETVAL_OK                          (0)


/**
 * This utility reads *.txt files from a path, specified as the only argument,
 * and builds a dictionary of words using a simplified Word2vec algorithm,
 * designed to be simple to understand rather than to be fast. Parameters may
 * be set with constants here above, but most of the times the defaults 
 * will be ok.
 *
 * Files are read line by line, considering each line a different text. 
 * Numbers are skipped.
 */
int main(int argc, const char * argv[]) {
	@autoreleasepool {
		if (argc != 2)
			return RETVAL_MISSING_ARGUMENT;
		
		// Get directory list at path of first argument
		NSFileManager *manager= [NSFileManager defaultManager];
		NSString *path= [[NSString alloc] initWithCString:argv[1] encoding:NSUTF8StringEncoding];
		NSArray *fileNames= [manager contentsOfDirectoryAtPath:path error:nil];
		
		// Prepare the dictionary
		MLWordDictionary *dictionary= [MLWordDictionary dictionaryWithMaxSize:300000];
		
		// First loop on all the files to determine the dictionary
		for (NSString *fileName in fileNames) {
			NSString *filePath= [path stringByAppendingPathComponent:fileName];
			
			// Skip non-txt files
			if (![fileName hasSuffix:@".txt"])
				continue;
			
			NSLog(@"Building dictionary with file: %@", fileName);

			IOLineReader *reader= [[IOLineReader alloc] initWithFilePath:filePath];

			do {
				NSString *line= [reader readLine];
				if (!line)
					break;
				
				// Fix residual HTML line breaks
				line= [line stringByReplacingOccurrencesOfString:@"<br />" withString:@" "];
				
				// Build the dictionary with the current line
				[MLBagOfWords buildDictionaryWithText:line
											 textID:nil
										 dictionary:dictionary
										   language:nil
									  wordExtractor:MLWordExtractorTypeSimpleTokenizer
								   extractorOptions:MLWordExtractorOptionOmitNumbers];
				
				if (reader.lineNumber % 1000 == 0)
					NSLog(@"- line: %6lu", reader.lineNumber);

			} while (YES);
			
			[reader close];
		}
		
		NSLog(@"Total number of words: %8lu", dictionary.totalWords);
		NSLog(@"Pre-filtering size:    %8lu", dictionary.size);
		
		// Filter dictionary for rare words
		[dictionary keepWordsWithHighestOccurrenciesUpToSize:DICTIONARY_SIZE];
		[dictionary compact];
		
		NSLog(@"Final dictionary size: %8lu", dictionary.size);
		NSLog(@"Dictionary:\n%@", dictionary);
		
		// Prepare the neural network:
		// - input and output sizes are set to the dictionary (bag of words) size
		// - hidden size is set to the desired vector size
		// - activation function is logistic
		MLNeuralNetwork *net= [[MLNeuralNetwork alloc] initWithLayerSizes:@[[NSNumber numberWithInt:(int) dictionary.size],
																			@VECTOR_SIZE,
																			[NSNumber numberWithInt:(int) dictionary.size]]
													   outputFunctionType:MLActivationFunctionTypeLogistic];
		
		// Loop all the files
		NSUInteger totalWords= 0;
		for (NSString *fileName in fileNames) {
			NSString *filePath= [path stringByAppendingPathComponent:fileName];

			// Skip non-txt files
			if (![fileName hasSuffix:@".txt"])
				continue;
			
			// Mark time
			NSDate *begin= [NSDate date];
			
			IOLineReader *reader= [[IOLineReader alloc] initWithFilePath:filePath];
			
			NSLog(@"Training word2vec with file: %@", fileName);

			do {
				@autoreleasepool {
					
					// Read next line
					NSString *line= [reader readLine];
					if (!line)
						break;
					
					// Fix residual HTML line breaks
					line= [line stringByReplacingOccurrencesOfString:@"<br />" withString:@" "];

					NSArray *words= [MLBagOfWords extractWordsWithSimpleTokenizerFromText:line
																			 withLanguage:nil
																		 extractorOptions:MLWordExtractorOptionOmitNumbers];
					
					// Pick a random starting point for the context
					NSUInteger start= [MLRandom nextUIntWithMax:SKIP_GRAM_WINDOW];

					// Build the context and pick up the central word
					NSMutableArray *context= [[NSMutableArray alloc] initWithCapacity:SKIP_GRAM_WINDOW *2];
					NSString *centralWord= nil;

					int pickedUpWords= 0;
					for (int i= 0; pickedUpWords < (SKIP_GRAM_WINDOW * 2) +1; i++) {
						
						// Check if we ran out of words for this line
						if ((start + i) >= words.count)
							break;
						
						// Pick the i-th word from the starting point
						NSString *word= [words objectAtIndex:start + i];
						
						// Skip the word if it's not in the dictionary
						MLWordInfo *wordInfo= [dictionary infoForWord:word];
						if (!wordInfo)
							continue;
						
						// Check if the word is too frequent: above the limit,
						// it is randomly downsampled
						double ranking= (sqrt(((double) wordInfo.totalOccurrencies) / (dictionary.totalWords * DOWNSAMPLING_FREQUENCY_LIMIT)) + 1.0) *
							((dictionary.totalWords * DOWNSAMPLING_FREQUENCY_LIMIT) / ((double) wordInfo.totalOccurrencies));

						if (ranking < 1.0) {
							double random= [MLRandom nextDouble];
							if (ranking < random)
								continue;
						}
						
						if (pickedUpWords == SKIP_GRAM_WINDOW)
							centralWord= word;
						else
							[context addObject:word];
						
						pickedUpWords++;
					}

					// Skip this line if the context is incomplete
					if (context.count < SKIP_GRAM_WINDOW * 2)
						continue;
					
					// Use bag of words to load the net's input buffer
					[MLBagOfWords bagOfWordsWithWords:@[centralWord]
											   textID:nil
										   dictionary:dictionary
									  buildDictionary:NO
								 featureNormalization:MLFeatureNormalizationTypeBoolean
										 outputBuffer:net.inputBuffer];
					
					// Use bag of words to load the net's expected output buffer
					[MLBagOfWords bagOfWordsWithWords:context
											   textID:nil
										   dictionary:dictionary
									  buildDictionary:NO
								 featureNormalization:MLFeatureNormalizationTypeBoolean
										 outputBuffer:net.expectedOutputBuffer];

					// Run the network
					[net feedForward];
					
					// Backpropagate the network
					[net backPropagateWithLearningRate:LEARNING_RATE];
					[net updateWeights];
					
					totalWords += words.count;
					NSTimeInterval elapsed= [[NSDate date] timeIntervalSinceDate:begin];
					
					if (reader.lineNumber % 1000 == 0)
						NSLog(@"- line: %6lu, words: %6luK, avg. speed: %6.2fK words/sec", reader.lineNumber, totalWords / 1000, (((double) totalWords) / elapsed) / 1000.0);
				}
				
			} while (YES);
			
			[reader close];
		}
		
		// !! TO DO: to be completed
	}
	
    return RETVAL_OK;
}

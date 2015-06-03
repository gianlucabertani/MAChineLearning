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

#define DICTIONARY_SIZE                (20000)
#define VECTOR_SIZE                      (100)
#define SKIP_GRAM_WINDOW                   (5)
#define DOWNSAMPLING_FREQUENCY_LIMIT       (0.001)
#define LEARNING_RATE                      (0.025)

#define RETVAL_MISSING_ARGUMENT            (9)
#define RETVAL_BUFFER_ALLOCATION_ERROR    (17)
#define RETVAL_OK                          (0)

// Uncomment to dump dictionary when ready
//#define DUMP_DICTIONARY


/**
 * This function scans a list of files and builds a word dictionary,
 * discarding numbers and keeping all the rest.
 */
MLWordDictionary *buildDictionary(NSArray *filePaths) {
	
	// Prepare the dictionary
	MLMutableWordDictionary *dictionary= [MLMutableWordDictionary dictionaryWithMaxSize:20 * DICTIONARY_SIZE];
	
	// Loop on all the files to build the dictionary
	for (NSString *filePath in filePaths) {
		@autoreleasepool {
			
			// Skip non-txt files
			if (![filePath hasSuffix:@".txt"])
				continue;
			
			NSLog(@"Building dictionary with file: %@", filePath);
			
			// Use the library's I/O line reader
			IOLineReader *reader= [[IOLineReader alloc] initWithFilePath:filePath];
			
			do {
				NSString *line= [reader readLine];
				if (!line)
					break;
				
				// Fix residual HTML line breaks (this is actually a specific flaw
				// of the text used for testing, but won't make any harm for other texts)
				line= [line stringByReplacingOccurrencesOfString:@"<br />" withString:@" "];
				
				// Build the dictionary with the current line
				[MLBagOfWords buildDictionaryWithText:line
											   documentID:filePath
										   dictionary:dictionary
											 language:nil
										wordExtractor:MLWordExtractorTypeSimpleTokenizer
									 extractorOptions:MLWordExtractorOptionOmitNumbers];
				
				if (reader.lineNumber % 1000 == 0)
					NSLog(@"- Lines read: %8lu, words: %7luK", reader.lineNumber, dictionary.totalWords / 1000);
				
			} while (YES);
			
			[reader close];
		}
	}
	
	NSLog(@"Total number of words:         %10lu", dictionary.totalWords);
	NSLog(@"Pre-filtering unique words:    %10lu", dictionary.size);
	
	// Filter dictionary and keep only most frequent words
	return [dictionary keepWordsWithHighestOccurrenciesUpToSize:DICTIONARY_SIZE];
}

/**
 * Word2vec main: builds a word dictionary based on the txt files found at the
 * specified path. Then trains a word2vec model using a neural network. Finally
 * tests the model and saves it.
 *
 * This implementation is designed to be understandable, not to be particularly
 * fast. Compared to reference implementation, this is monothreaded and makes no
 * particular optimization, but takes advantage of neural network vectorization.
 * Expect it to be faster with small dictionaries and way slower with large
 * dictionaries.
 */
int main(int argc, const char * argv[]) {
	@autoreleasepool {
		if (argc != 2)
			return RETVAL_MISSING_ARGUMENT;

		// Preprare the path from the first argument
		NSString *path= [[NSString alloc] initWithCString:argv[1] encoding:NSUTF8StringEncoding];
		
		// Get directory list at path
		NSFileManager *manager= [NSFileManager defaultManager];
		NSArray *fileNames= [manager contentsOfDirectoryAtPath:path error:nil];
		
		// Preapre a list of file paths
		NSMutableArray *filePaths= [[NSMutableArray alloc] init];
		for (NSString *fileName in fileNames)
			[filePaths addObject:[path stringByAppendingPathComponent:fileName]];
		
		// Build the dictionary
		MLWordDictionary *dictionary= buildDictionary(filePaths);
		
		NSLog(@"Final unique words:            %10lu", dictionary.size);
		
#ifdef DUMP_DICTIONARY
		NSLog(@"Dictionary:\n%@", dictionary);
#endif // DUMP_DICTIONARY
		
		// Prepare the neural network:
		// - input and output sizes are set to the dictionary (bag of words) size
		// - hidden size is set to the desired vector size
		// - activation function is logistic
		MLNeuralNetwork *net= [[MLNeuralNetwork alloc] initWithLayerSizes:@[[NSNumber numberWithUnsignedInteger:dictionary.size],
																			@VECTOR_SIZE,
																			[NSNumber numberWithUnsignedInteger:dictionary.size]]
													   outputFunctionType:MLActivationFunctionTypeLogistic];
		
		// Loop all the files
		NSUInteger totalWords= 0;
		for (NSString *filePath in filePaths) {

			// Skip non-txt files
			if (![filePath hasSuffix:@".txt"])
				continue;
			
			// Mark time
			NSDate *begin= [NSDate date];
			
			IOLineReader *reader= [[IOLineReader alloc] initWithFilePath:filePath];
			
			NSLog(@"Training word2vec with file: %@", filePath);

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
											   documentID:nil
										   dictionary:dictionary
									  buildDictionary:NO
								 featureNormalization:MLFeatureNormalizationTypeBoolean
										 outputBuffer:net.inputBuffer];
					
					// Use bag of words to load the net's expected output buffer
					[MLBagOfWords bagOfWordsWithWords:context
											   documentID:nil
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
					
					if (reader.lineNumber % 100 == 0)
						NSLog(@"- Lines trained: %8lu, words: %7luK, avg. speed: %6.2fK words/sec", reader.lineNumber, totalWords / 1000, (((double) totalWords) / elapsed) / 1000.0);
				}
				
			} while (YES);
			
			[reader close];
		}
		
		// !! TO DO: to be completed
	}
	
    return RETVAL_OK;
}

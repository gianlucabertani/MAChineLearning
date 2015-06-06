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
#import <Accelerate/Accelerate.h>

#define DICTIONARY_SIZE                 (5000)
#define VECTOR_SIZE                      (100)
#define CONTEXT_WINDOW                     (5)
#define LEARNING_RATE                      (0.025)
#define TRAIN_CYCLES                       (5)

#define RETVAL_MISSING_ARGUMENT            (9)
#define RETVAL_BUFFER_ALLOCATION_ERROR    (17)
#define RETVAL_OK                          (0)

// Uncomment to dump dictionary
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
											 language:@"en"
										wordExtractor:MLWordExtractorTypeSimpleTokenizer
									 extractorOptions:MLWordExtractorOptionOmitStopWords | MLWordExtractorOptionOmitNumbers];
				
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
 * Tests the model by subtracting and adding two vectors from a base vector,
 * and checking the distance from the expect resulting word.
 */
BOOL testModel(MLWordVectorMap *map, NSString *baseWord, NSString *minusWord, NSString *plusWord, NSString *expectWord) {
	MLWordVector *base= [map vectorForWord:baseWord];
	MLWordVector *minus= [map vectorForWord:minusWord];
	MLWordVector *plus= [map vectorForWord:plusWord];
	MLWordVector *result= [[base subtractVector:minus] addVector:plus];
	
	NSArray *wordsByDistance= [map nearestWordsToVector:result];
	BOOL expectedIsThere= NO;
	
	NSMutableString *words= [[NSMutableString alloc] init];
	for (int i= 0; i < 3; i++) {
		NSString *word= [wordsByDistance objectAtIndex:i];
		expectedIsThere= expectedIsThere || [word isEqualToString:expectWord];
		
		if (i > 0)
			[words appendString:@", "];
		
		[words appendString:word];
	}
	
	if (!expectedIsThere) {
		NSUInteger pos= [wordsByDistance indexOfObject:expectWord];
		[words appendFormat:@"... %@ (%lu)", expectWord, pos];
	}
	
	NSLog(@"Testing model: %@ -%@ +%@ = %@", baseWord, minusWord, plusWord, words);
	
	return expectedIsThere;
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
		
		// Prepare the buffer for computing the error
		MLReal *errorBuffer= NULL;
		int err= posix_memalign((void **) &errorBuffer,
								BUFFER_MEMORY_ALIGNMENT,
								sizeof(MLReal) * dictionary.size);
		if (err) {
			NSLog(@"Error while allocating error buffer: %d", err);
			return RETVAL_BUFFER_ALLOCATION_ERROR;
		}

		// Loop for train cicles
		NSUInteger trainingCycles= 0;
		NSUInteger totalWords= 0;
		do {
		
			// Stat counters
			NSUInteger partialWords= 0;
			MLReal avgError= 0.0;
			NSDate *begin= nil;
			
			// Loop all the files
			for (NSString *filePath in filePaths) {

				// Skip non-txt files
				if (![filePath hasSuffix:@".txt"])
					continue;
				
				IOLineReader *reader= [[IOLineReader alloc] initWithFilePath:filePath];
				
				NSLog(@"Training cycle %lu with file: %@", trainingCycles +1, filePath);
				
				do {
					@autoreleasepool {
						
						// Mark time
						if (!begin)
							begin= [NSDate date];
						
						// Read next line
						NSString *line= [reader readLine];
						if (!line)
							break;
						
						// Fix residual HTML line breaks
						line= [line stringByReplacingOccurrencesOfString:@"<br />" withString:@" "];

						NSArray *words= [MLBagOfWords extractWordsWithSimpleTokenizerFromText:line
																				 withLanguage:@"en"
																			 extractorOptions:MLWordExtractorOptionOmitStopWords | MLWordExtractorOptionOmitNumbers];
						
						// Loop for all possible windows
						NSUInteger offset= 0;
						do {
						
							// Pick a random starting point for the context
							NSUInteger start= offset + [MLRandom nextUIntWithMax:CONTEXT_WINDOW];

							// Build the context and pick up the central word
							NSMutableArray *context= [[NSMutableArray alloc] initWithCapacity:CONTEXT_WINDOW *2];
							NSString *centralWord= nil;

							int i= 0, pickedUpWords= 0;
							for (; pickedUpWords < (CONTEXT_WINDOW * 2) +1; i++) {
								
								// Check if we ran out of words for this line
								if ((start + i) >= words.count)
									break;
								
								// Pick the i-th word from the starting point
								NSString *word= [words objectAtIndex:start + i];
								
								// Skip the word if it's not in the dictionary
								MLWordInfo *wordInfo= [dictionary infoForWord:word];
								if (!wordInfo)
									continue;
								
								if (pickedUpWords == CONTEXT_WINDOW)
									centralWord= word;
								else
									[context addObject:word];
								
								pickedUpWords++;
							}
							
							// Exit the loop when the context is incomplete
							if (context.count < CONTEXT_WINDOW * 2)
								break;
							
							offset += i;
							
							// Use bag of words to load the net's input buffer
							[MLBagOfWords bagOfWordsWithWords:@[centralWord]
												   documentID:nil
												   dictionary:dictionary
											  buildDictionary:NO
										 featureNormalization:MLFeatureNormalizationTypeBoolean
												 outputBuffer:net.inputBuffer];
							
							// Run the network
							[net feedForward];
							
							// Use bag of words to load the net's expected output buffer
							[MLBagOfWords bagOfWordsWithWords:context
												   documentID:nil
												   dictionary:dictionary
											  buildDictionary:NO
										 featureNormalization:MLFeatureNormalizationTypeL2
												 outputBuffer:net.expectedOutputBuffer];
							
							// Compute the error (for statistics only)
							ML_VDSP_VSUB(net.expectedOutputBuffer, 1, net.outputBuffer, 1, errorBuffer, 1, dictionary.size);

							MLReal error= 0.0;
							ML_VDSP_SVESQ(errorBuffer, 1, &error, dictionary.size);
							avgError += error / 2.0;
							
							// Compute the current learning rate
							MLReal learningRate= LEARNING_RATE * (1.0 - (totalWords / (1.0 + TRAIN_CYCLES * dictionary.totalWords)));
							if (learningRate < LEARNING_RATE * 0.0001)
								learningRate= LEARNING_RATE * 0.0001;
							
							// Backpropagate the network
							[net backPropagateWithLearningRate:learningRate];
							[net updateWeights];
							
						} while (YES);
						
						totalWords += words.count;
						partialWords += words.count;
						
						NSTimeInterval elapsed= [[NSDate date] timeIntervalSinceDate:begin];
						
						if (reader.lineNumber % 100 == 0) {
							avgError /= 100.0;
							
							NSLog(@"- Lines trained: %8lu, words: %7luK, speed: %6.2fK w/s, error: %5.2f", reader.lineNumber, totalWords / 1000, (((double) partialWords) / elapsed) / 1000.0, avgError);

							avgError= 0.0;
							partialWords= 0;
							begin= nil;
						}
						
						if (reader.lineNumber % 1000 == 0) {
							
							// Test the model
							MLWordVectorMap *map= [[MLWordVectorMap alloc] initWithWord2vecNeuralNet:net dictionary:dictionary];

							// Present-past
							testModel(map, @"watched", @"watching", @"going", @"went");
							
							// Male-female
							testModel(map, @"king", @"man", @"woman", @"queen");
							
							// Singluar-plural
							testModel(map, @"children", @"child", @"hand", @"hands");
							
							// Noun-adjective
							testModel(map, @"french", @"france", @"america", @"american");
							
							// Nation-capital
							testModel(map, @"paris", @"france", @"england", @"london");
						}
					}
					
				} while (YES);
				
				[reader close];
			}
			
			trainingCycles++;
			
		} while (trainingCycles < TRAIN_CYCLES);
		
		// !! TO DO: to be completed
	}
	
    return RETVAL_OK;
}

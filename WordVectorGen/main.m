//
//  main.m
//  WordVectorGen
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
#define CONTEXT_WINDOW                     (6)
#define LEARNING_RATE                      (0.05)
#define MIN_LEARNING_RATE                  (0.00005)
#define TRAIN_CYCLES                       (1)

#define RETVAL_MISSING_ARGUMENT            (9)
#define RETVAL_BUFFER_ALLOCATION_ERROR    (17)
#define RETVAL_OK                          (0)


// Forwards
MLWordDictionary *buildDictionary(NSString *textPath, void(^)(NSUInteger line, MLWordDictionary *dictionary));
NSArray *buildEquivalenceList(NSString *equivalenceListPath, MLWordDictionary *dictionary);
BOOL extractContext(MLWordDictionary *dictionary, NSArray *words, NSUInteger offset, NSMutableArray *context, NSMutableString *centralWord);
MLReal testModel(MLWordVectorMap *map, NSArray *equivalenceList);


/**
 * WordVectorGen main: builds a word dictionary based on the txt files found at the
 * specified path. Then trains a word-vector model using a neural network. Finally
 * tests the model and saves it.
 *
 * This implementation is designed to be understandable, not to be particularly
 * fast. Compared to other implementations this is monothreaded and makes no
 * particular optimization, but takes advantage of neural network vectorization.
 * Expect it to be relatively fast with a small dictionary and very slow with a
 * large dictionary.
 */
int main(int argc, const char * argv[]) {
	@autoreleasepool {
		if (argc != 4)
			return RETVAL_MISSING_ARGUMENT;

		// Prepare file paths from arguments
		NSString *textPath= [[NSString alloc] initWithCString:argv[1] encoding:NSUTF8StringEncoding];
		NSString *equivalenceListPath= [[NSString alloc] initWithCString:argv[2] encoding:NSUTF8StringEncoding];
		NSString *comparisonModelPath= [[NSString alloc] initWithCString:argv[3] encoding:NSUTF8StringEncoding];
		
		// Build the dictionary
		NSLog(@"Building dictionary with file: %@", textPath);
		
		__block NSUInteger totLines= 0;
		MLWordDictionary *fullDictionary= buildDictionary(textPath, ^(NSUInteger linesRead, MLWordDictionary *dictionary) {
			totLines= linesRead;

			if (linesRead % 1000 == 0)
				NSLog(@"- Lines read: %8lu, words: %7luK", linesRead, dictionary.totalWords / 1000);
		});
		
		NSLog(@"Total number of lines:         %10lu", totLines);
		NSLog(@"Total number of words:         %10lu", fullDictionary.totalWords);
		NSLog(@"Pre-filtering unique words:    %10lu", fullDictionary.size);
		
		// Filter dictionary and keep only most frequent words
		MLWordDictionary *dictionary= [fullDictionary keepWordsWithHighestOccurrenciesUpToSize:DICTIONARY_SIZE];

		NSLog(@"Final unique words:            %10lu", dictionary.size);
		NSLog(@"Final dictionary: %@", dictionary);

		// Build equivalence list
		NSLog(@"Building equivalence list with file: %@", equivalenceListPath);

		NSArray *equivalenceList= buildEquivalenceList(equivalenceListPath, dictionary);
		
		NSLog(@"Number of equivalences:                %5lu", equivalenceList.count);
		
		// Test the comparison model
		MLWordVectorMap *compMap= [MLWordVectorMap createFromWord2vecFile:comparisonModelPath binary:YES];
		MLReal compEquivalenceScore= testModel(compMap, equivalenceList);
		
		NSLog(@"Comparison model: score of %8.2f on %5lu equivalences or %5.2f%%", compEquivalenceScore, equivalenceList.count, 100.0 * (compEquivalenceScore / ((float) equivalenceList.count)));
		
		// Prepare the neural network:
		// - input and output sizes are set to the dictionary (bag of words) size
		// - hidden size is set to the desired vector size
		// - activation function is logistic
		MLNeuralNetwork *net= [[MLNeuralNetwork alloc] initWithLayerSizes:@[[NSNumber numberWithUnsignedInteger:dictionary.size],
																			@VECTOR_SIZE,
																			[NSNumber numberWithUnsignedInteger:dictionary.size]]
																  useBias:NO
														 costFunctionType:MLCostFunctionTypeCrossEntropy
													  backPropagationType:MLBackPropagationTypeStandard
													   hiddenFunctionType:MLActivationFunctionTypeLogistic
													   outputFunctionType:MLActivationFunctionTypeLogistic];
		
		// Randomization of network weights (i.e. initial vectors)
		[net randomizeWeights];
		
		// Prepare the buffer for computing the error
		MLReal *errorBuffer= NULL;
		int err= posix_memalign((void **) &errorBuffer,
								BUFFER_MEMORY_ALIGNMENT,
								sizeof(MLReal) * dictionary.size);
		if (err) {
			NSLog(@"Error while allocating error buffer: %d", err);
			return RETVAL_BUFFER_ALLOCATION_ERROR;
		}

		// Global stat counters
		NSUInteger trainingCycles= 0;
		NSUInteger linesRead= 0;
		NSDate *totalBegin= [NSDate date];
		MLReal progress= 0.0;
		
		// Loop for train cicles
		do {
		
			// Partial stat counters (they are reset every X lines)
			NSUInteger scannedContexts= 0;
			MLReal avgError= 0.0;
			NSDate *blockBegin= nil;
			
			// Use the library's I/O line reader
			IOLineReader *reader= [[IOLineReader alloc] initWithFilePath:textPath];
			
			NSLog(@"Training cycle %lu with file: %@", trainingCycles +1, textPath);
			
			do {
				@autoreleasepool {
					
					// Mark time
					if (!blockBegin)
						blockBegin= [NSDate date];
					
					// Read next line
					NSString *line= [reader readLine];
					if (!line)
						break;
					
					// Fix residual HTML line breaks
					line= [line stringByReplacingOccurrencesOfString:@"<br />" withString:@" "];

					// Extract words from line
					NSArray *words= [MLBagOfWords extractWordsWithSimpleTokenizerFromText:line
																			 withLanguage:@"en"
																		 extractorOptions:MLWordExtractorOptionOmitNumbers];
					
					// Extract the context, and skip the line if
					// the context is incomplete
					NSMutableArray *context= [[NSMutableArray alloc] initWithCapacity:CONTEXT_WINDOW * 2];
					NSMutableString *centralWord= [[NSMutableString alloc] init];
					NSUInteger offset= 0;

					// Loop for all possible contexts in the line
					do {
						BOOL complete= extractContext(dictionary, words, offset, context, centralWord);
						if (!complete)
							break;
					
						// Use bag of words to load the net's input buffer with
						// the context
						[MLBagOfWords bagOfWordsWithWords:context
											   documentID:nil
											   dictionary:dictionary
										  buildDictionary:NO
									 featureNormalization:MLFeatureNormalizationTypeL2
											 outputBuffer:net.inputBuffer];
						
						// Run the network
						[net feedForward];
						
						// Use bag of words to load the net's expected output buffer
						// with the central word
						[MLBagOfWords bagOfWordsWithWords:@[centralWord]
											   documentID:nil
											   dictionary:dictionary
										  buildDictionary:NO
									 featureNormalization:MLFeatureNormalizationTypeBoolean
											 outputBuffer:net.expectedOutputBuffer];
						
						// Compute the error (for statistics only)
						avgError += net.cost;
						
						// Compute the current learning rate
						MLReal learningRate= MAX(LEARNING_RATE * (1.0 - progress), MIN_LEARNING_RATE);
						
						// Backpropagate the network
						[net backPropagateWithLearningRate:learningRate];
						[net updateWeights];
						
						offset++;
						
						// Update statistics
						scannedContexts++;
					
					} while (YES);

					// Update and log some statistics
					linesRead++;
					progress= ((MLReal) linesRead) / ((MLReal) TRAIN_CYCLES * totLines);
					
					if (reader.lineNumber % 10 == 0) {
						avgError /= (MLReal) scannedContexts;

						NSTimeInterval totalETA= ([[NSDate date] timeIntervalSinceDate:totalBegin] / progress) * (1.0 - progress);
						NSTimeInterval blockElapsed= [[NSDate date] timeIntervalSinceDate:blockBegin];
						NSString *etaTime= [[[NSDateComponentsFormatter alloc] init] stringFromTimeInterval:totalETA];
						
						NSLog(@"- Lines read: %8lu, cycle: %lu, prog.: %5.2f%%, speed: %7.2f w/s, cost: %6.2f, ETA: %@", reader.lineNumber, trainingCycles +1, 100.0 * progress, (((double) scannedContexts) / blockElapsed), avgError, etaTime);

						avgError= 0.0;
						scannedContexts= 0;
						blockBegin= nil;
					}
					
					if (reader.lineNumber % 10000 == 0) {
						
						// Test the model
						MLWordVectorMap *map= [MLWordVectorMap createFromNeuralNetwork:net dictionary:dictionary];
						MLReal equivalenceScore= testModel(map, equivalenceList);
						
						NSLog(@"Testing model:    score of %8.2f on %5lu equivalences or %5.2f%%", equivalenceScore, equivalenceList.count, 100.0 * (equivalenceScore / ((float) equivalenceList.count)));
					}
				}
				
			} while (YES);
			
			[reader close];
			
			trainingCycles++;
			
		} while (trainingCycles < TRAIN_CYCLES);
		
		// Final test of the model, reporting also again
		// the results of the comparison model
		MLWordVectorMap *map= [MLWordVectorMap createFromNeuralNetwork:net dictionary:dictionary];
		MLReal equivalenceScore= testModel(map, equivalenceList);
		
		NSLog(@"Testing model:    score of %8.2f on %5lu equivalences or %5.2f%%", equivalenceScore, equivalenceList.count, 100.0 * (equivalenceScore / ((float) equivalenceList.count)));
		NSLog(@"Comparison model: score of %8.2f on %5lu equivalences or %5.2f%%", compEquivalenceScore, equivalenceList.count, 100.0 * (compEquivalenceScore / ((float) equivalenceList.count)));
	}
	
    return RETVAL_OK;
}


/**
 * This function scans a text file and builds a word dictionary,
 * discarding numbers and keeping all the rest.
 */
MLWordDictionary *buildDictionary(NSString *textPath, void(^lineReadHandler)(NSUInteger linesRead, MLWordDictionary *dictionary)) {
	
	// Prepare the dictionary
	MLMutableWordDictionary *dictionary= [MLMutableWordDictionary dictionaryWithMaxSize:20 * DICTIONARY_SIZE];
	
	@autoreleasepool {
		
		// Use the library's I/O line reader
		IOLineReader *reader= [[IOLineReader alloc] initWithFilePath:textPath];
		
		do {
			NSString *line= [reader readLine];
			if (!line)
				break;
			
			// Fix residual HTML line breaks (this is actually a specific flaw
			// of the text used for testing, but won't make any harm for other texts)
			line= [line stringByReplacingOccurrencesOfString:@"<br />" withString:@" "];
			
			// Build the dictionary with the current line
			[MLBagOfWords buildDictionaryWithText:line
									   documentID:textPath
									   dictionary:dictionary
										 language:@"en"
									wordExtractor:MLWordExtractorTypeSimpleTokenizer
								 extractorOptions:MLWordExtractorOptionOmitNumbers];
			
			lineReadHandler(reader.lineNumber, dictionary);
			
		} while (YES);
		
		[reader close];
	}
	
	return dictionary;
}


/**
 * This function scans a text file for word equivalences in the form:<br/>
 * word1 word2 word3 word4
 *
 * Where the equivalence is interpreted in this way:<br/>
 * word1 : word2 = word3 : word4
 *
 * E.g.:<br/>
 * looking looked reading read
 */
NSArray *buildEquivalenceList(NSString *equivalenceListPath, MLWordDictionary *dictionary) {
	
	// Prepare the list
	NSMutableArray *equivalenceList= [[NSMutableArray alloc] init];
	
	@autoreleasepool {
		
		// Use the library's I/O line reader
		IOLineReader *reader= [[IOLineReader alloc] initWithFilePath:equivalenceListPath];
		
		do {
			NSString *line= [reader readLine];
			if (!line)
				break;
			
			NSArray *words= [line componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
			
			// Skip non-equivalence lines
			if (words.count != 4)
				continue;
			
			// Extract
			NSString *baseWord= [[words objectAtIndex:1] stringByTrimmingCharactersInSet:[NSCharacterSet newlineCharacterSet]];
			NSString *minusWord= [[words objectAtIndex:0] stringByTrimmingCharactersInSet:[NSCharacterSet newlineCharacterSet]];
			NSString *plusWord= [[words objectAtIndex:2] stringByTrimmingCharactersInSet:[NSCharacterSet newlineCharacterSet]];
			NSString *expectedWord= [[words objectAtIndex:3] stringByTrimmingCharactersInSet:[NSCharacterSet newlineCharacterSet]];
			
			// Skip comment lines
			if ([minusWord hasPrefix:@":"])
				continue;
			
			// Add equivalence only if all words are present
			// in the dictionary
			if ([dictionary containsWord:baseWord] &&
				[dictionary containsWord:minusWord] &&
				[dictionary containsWord:plusWord] &&
				[dictionary containsWord:expectedWord]) {
				
				[equivalenceList addObject:@{@"baseWord": baseWord,
											 @"minusWord": minusWord,
											 @"plusWord": plusWord,
											 @"expectedWord": expectedWord}];
			}
			
		} while (YES);
		
		[reader close];
	}
	
	return equivalenceList;
}


/**
 * This function scans a word list and builds a context window,
 * also finding the central word of the context. Returns NO
 * if the context is incomplete.
 */
BOOL extractContext(MLWordDictionary *dictionary, NSArray *words, NSUInteger offset, NSMutableArray *context, NSMutableString *centralWord) {
	
	// Clear the context and central word
	[context removeAllObjects];
	[centralWord setString:@""];
	
	int i= 0, pickedUpWords= 0;
	for (; pickedUpWords < (CONTEXT_WINDOW * 2) +1; i++) {
		
		// Check if we ran out of words for this line
		if ((offset + i) >= words.count)
			break;
		
		// Pick the i-th word from the starting point
		NSString *word= [words objectAtIndex:offset + i];
		
		// Skip the word if it's not in the dictionary
		MLWordInfo *wordInfo= [dictionary infoForWord:word];
		if (!wordInfo)
			continue;
		
		if (pickedUpWords == CONTEXT_WINDOW)
			[centralWord setString:word];
		else
			[context addObject:word];
		
		pickedUpWords++;
	}
	
	return (context.count >= CONTEXT_WINDOW * 2);
}


/**
 * Tests the model by subtracting and adding two vectors from a base vector,
 * and checking the distance from the expect resulting word.
 */
MLReal testModel(MLWordVectorMap *map, NSArray *equivalenceList) {
	
	// Prepare counters
	MLReal score= 0.0;
	
	@autoreleasepool {
		
		// Loop on all equivalences
		for (NSDictionary *equivalence in equivalenceList) {
			NSString *baseWord= [equivalence objectForKey:@"baseWord"];
			NSString *minusWord= [equivalence objectForKey:@"minusWord"];
			NSString *plusWord= [equivalence objectForKey:@"plusWord"];
			NSString *expectedWord= [equivalence objectForKey:@"expectedWord"];
			
			// Test the equivalence
			MLWordVector *base= [map vectorForWord:baseWord];
			MLWordVector *minus= [map vectorForWord:minusWord];
			MLWordVector *plus= [map vectorForWord:plusWord];
			
			if ((!base) || (!minus) || (!plus))
				continue;
			
			MLWordVector *result= [[base subtractVector:minus] addVector:plus];
			NSArray *nearestWords= [map nearestWordsToVector:result];
			
			if ([[nearestWords objectAtIndex:0] caseInsensitiveCompare:expectedWord] == NSOrderedSame) {
				score += 1.0;
				
			} else if ([[nearestWords objectAtIndex:1] caseInsensitiveCompare:expectedWord] == NSOrderedSame) {
				score += 0.5;
			
			} else if ([[nearestWords objectAtIndex:2] caseInsensitiveCompare:expectedWord] == NSOrderedSame) {
				score += 0.25;
			}
		}
	}
	
	return score;
}


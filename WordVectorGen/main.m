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
#define TRAIN_CYCLES                       (5)

#define RETVAL_MISSING_ARGUMENT            (9)
#define RETVAL_BUFFER_ALLOCATION_ERROR    (17)
#define RETVAL_OK                          (0)

// Uncomment to dump dictionary
//#define DUMP_DICTIONARY


// Forwards
MLWordDictionary *buildDictionary(NSString *textPath);
NSArray *buildEquivalenceList(NSString *equivalenceListPath, MLWordDictionary *dictionary);
BOOL extractContext(MLWordDictionary *dictionary, NSArray *words, NSUInteger offset, NSMutableArray *context, NSMutableString *centralWord);
void testModel(MLWordVectorMap *map, NSArray *equivalenceList);


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
		if (argc != 3)
			return RETVAL_MISSING_ARGUMENT;

		// Prepare file paths from arguments
		NSString *textPath= [[NSString alloc] initWithCString:argv[1] encoding:NSUTF8StringEncoding];
		NSString *equivalenceListPath= [[NSString alloc] initWithCString:argv[2] encoding:NSUTF8StringEncoding];
		
		// Build the dictionary
		MLWordDictionary *dictionary= buildDictionary(textPath);
		
		NSLog(@"Final unique words:            %10lu", dictionary.size);
		
#ifdef DUMP_DICTIONARY
		NSLog(@"Dictionary:\n%@", dictionary);
#endif // DUMP_DICTIONARY

		// Build equivalence list
		NSArray *equivalenceList= buildEquivalenceList(equivalenceListPath, dictionary);
		
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
		MLReal progress= 0.0;
		do {
		
			// Stat counters
			NSUInteger partialWords= 0;
			MLReal avgError= 0.0;
			NSDate *begin= nil;
			
			// Use the library's I/O line reader
			IOLineReader *reader= [[IOLineReader alloc] initWithFilePath:textPath];
			
			NSLog(@"Training cycle %lu with file: %@", trainingCycles +1, textPath);
			
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

					// Extract words from line
					NSArray *words= [MLBagOfWords extractWordsWithSimpleTokenizerFromText:line
																			 withLanguage:@"en"
																		 extractorOptions:MLWordExtractorOptionOmitStopWords | MLWordExtractorOptionOmitNumbers];
					
					// Extract the context, and skip the line if
					// the context is incomplete
					NSMutableArray *context= [[NSMutableArray alloc] initWithCapacity:CONTEXT_WINDOW * 2];
					NSMutableString *centralWord= [[NSMutableString alloc] init];
					
					BOOL complete= extractContext(dictionary, words, 0, context, centralWord);
					if (!complete)
						continue;
					
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
					ML_VDSP_VSUB(net.expectedOutputBuffer, 1, net.outputBuffer, 1, errorBuffer, 1, dictionary.size);

					MLReal error= 0.0;
					ML_VDSP_SVESQ(errorBuffer, 1, &error, dictionary.size);
					avgError += error / 2.0;
					
					// Compute the current learning rate
					MLReal learningRate= MAX(LEARNING_RATE * (1.0 - progress), MIN_LEARNING_RATE);
					
					// Backpropagate the network
					[net backPropagateWithLearningRate:learningRate];
					[net updateWeights];

					// Update statistics
					totalWords += words.count;
					partialWords += words.count;
					progress= ((MLReal) totalWords) / ((MLReal) TRAIN_CYCLES * dictionary.totalWords);
					
					if (reader.lineNumber % 100 == 0) {
						NSTimeInterval elapsed= [[NSDate date] timeIntervalSinceDate:begin];
						avgError /= 100.0;
						
						NSLog(@"- Lines read: %8lu, words: %7luK, progress: %5.2f%%, speed: %6.2fK w/s, error: %5.2f", reader.lineNumber, totalWords / 1000, 100.0 * progress, (((double) partialWords) / elapsed) / 1000.0, avgError);

						avgError= 0.0;
						partialWords= 0;
						begin= nil;
					}
					
					if (reader.lineNumber % 5000 == 0) {
						
						// Test the model
						MLWordVectorMap *map= [[MLWordVectorMap alloc] initWithNeuralNetwork:net dictionary:dictionary];
						testModel(map, equivalenceList);
					}
				}
				
			} while (YES);
			
			[reader close];
			
			trainingCycles++;
			
		} while (trainingCycles < TRAIN_CYCLES);
		
		// Final test of the model
		MLWordVectorMap *map= [[MLWordVectorMap alloc] initWithNeuralNetwork:net dictionary:dictionary];
		testModel(map, equivalenceList);
	}
	
    return RETVAL_OK;
}


/**
 * This function scans a text file and builds a word dictionary,
 * discarding numbers and stop words and keeping all the rest.
 */
MLWordDictionary *buildDictionary(NSString *textPath) {
	
	// Prepare the dictionary
	MLMutableWordDictionary *dictionary= [MLMutableWordDictionary dictionaryWithMaxSize:20 * DICTIONARY_SIZE];
	
	@autoreleasepool {
		
		NSLog(@"Building dictionary with file: %@", textPath);
		
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
								 extractorOptions:MLWordExtractorOptionOmitStopWords | MLWordExtractorOptionOmitNumbers];
			
			if (reader.lineNumber % 1000 == 0)
				NSLog(@"- Lines read: %8lu, words: %7luK", reader.lineNumber, dictionary.totalWords / 1000);
			
		} while (YES);
		
		[reader close];
	}
	
	NSLog(@"Total number of words:         %10lu", dictionary.totalWords);
	NSLog(@"Pre-filtering unique words:    %10lu", dictionary.size);
	
	// Filter dictionary and keep only most frequent words
	return [dictionary keepWordsWithHighestOccurrenciesUpToSize:DICTIONARY_SIZE];
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
		
		NSLog(@"Building equivalence list with file: %@", equivalenceListPath);
		
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
	
	NSLog(@"Number of equivalences:        %10lu", equivalenceList.count);
	
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
	
	// Pick a random starting point for the context
	NSUInteger start= offset + [MLRandom nextUIntWithMax:CONTEXT_WINDOW];
	
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
void testModel(MLWordVectorMap *map, NSArray *equivalenceList) {
	
	// Prepare counters
	NSUInteger matchedEquivalences= 0;
	
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
			MLWordVector *result= [[base subtractVector:minus] addVector:plus];
			
			NSString *resultWord= [map nearestWordToVector:result];
			
			if ([expectedWord caseInsensitiveCompare:resultWord] == NSOrderedSame)
				matchedEquivalences++;
		}
	}
	
	NSLog(@"Testing model: matched %lu/%lu equivalences (%.2f%%)", matchedEquivalences, equivalenceList.count, 100.0 * ((float) matchedEquivalences) / ((float) equivalenceList.count));
}


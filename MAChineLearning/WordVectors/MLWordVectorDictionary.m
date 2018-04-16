//
//  MLWordVectorDictionary.m
//  MAChineLearning
//
//  Created by Gianluca Bertani on 03/06/15.
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

#import "MLWordVectorDictionary.h"
#import "MLWordVector.h"
#import "MLWordVectorException.h"
#import "MLWordDictionary.h"
#import "MLWordInfo.h"
#import "MLNeuralNetwork.h"
#import "MLNeuronLayer.h"
#import "MLNeuron.h"

#import "MLAlloc.h"

#import "IOLineReader.h"

#define WORD2VEC_MAX_WORD_LENGTH            (200)


#pragma mark -
#pragma mark MLWordVectorDictionary extension

@interface MLWordVectorDictionary () {
    NSUInteger _wordCount;
    NSUInteger _vectorSize;
    
	NSMutableDictionary *_vectors;
}

@end


#pragma mark -
#pragma mark MLWordVectorDictionary implementation

@implementation MLWordVectorDictionary


#pragma mark -
#pragma mark Initialization

+ (MLWordVectorDictionary *) createFromWord2vecFile:(NSString *)vectorFilePath binary:(BOOL)binary {
	
	// Checks
	NSFileManager *fileManger= [NSFileManager defaultManager];
	if (![fileManger fileExistsAtPath:vectorFilePath])
		@throw [MLWordVectorException wordVectorExceptionWithReason:@"File does not exist"
														   userInfo:@{@"filePath": vectorFilePath}];
	
	// Use ANSI C APIs to read the file, to ensure compatibility
	const char *filePath= [vectorFilePath cStringUsingEncoding:NSUTF8StringEncoding];
	
	FILE *f= fopen(filePath, "r");
	if (!f)
		@throw [MLWordVectorException wordVectorExceptionWithReason:@"File access denied"
														   userInfo:@{@"filePath": vectorFilePath}];
	NSMutableDictionary *vectorDictionary= nil;

	@try {
		int result= 0;
		
		NSUInteger dictionarySize= 0;
		NSUInteger vectorSize= 0;

		result= fscanf(f, "%lu", &dictionarySize);
		if (result != 1)
			@throw [MLWordVectorException wordVectorExceptionWithReason:@"Error while reading the dictionary size"
															   userInfo:@{@"result": [NSNumber numberWithInt:result]}];
		
		result= fscanf(f, "%lu", &vectorSize);
		if (result != 1)
			@throw [MLWordVectorException wordVectorExceptionWithReason:@"Error while reading the vector size"
															   userInfo:@{@"result": [NSNumber numberWithInt:result]}];
		
		// Prepare the transitory dictionary
		vectorDictionary= [[NSMutableDictionary alloc] initWithCapacity:dictionarySize];

		// Loop for all the words
		for (NSUInteger i= 0; i < dictionarySize; i++) {
			
			// Read the word
			char wordStr[WORD2VEC_MAX_WORD_LENGTH];
			result= fscanf(f, "%s ", wordStr);
			if (result != 1)
				@throw [MLWordVectorException wordVectorExceptionWithReason:@"Error while reading the next word"
																   userInfo:@{@"result": [NSNumber numberWithInt:result]}];
			
			// Prepare the vector
			NSMutableArray *vector= [[NSMutableArray alloc] initWithCapacity:vectorSize];
			
			for (NSUInteger j= 0; j < vectorSize; j++) {
				
				// Read and store the vector element
				if (binary) {
					float elem= 0.0;

					result= (int) fread(&elem, sizeof(float), 1, f);
					if (result != 1)
						@throw [MLWordVectorException wordVectorExceptionWithReason:@"Error while reading a vector element"
																		   userInfo:@{@"result": [NSNumber numberWithInt:result]}];
					
					[vector addObject:[NSNumber numberWithFloat:elem]];
					
				} else {
					double elem= 0.0;
					
					result= fscanf(f, "%lf", &elem);
					if (result != 1)
						@throw [MLWordVectorException wordVectorExceptionWithReason:@"Error while reading a vector element"
																		   userInfo:@{@"result": [NSNumber numberWithInt:result]}];
					
					[vector addObject:[NSNumber numberWithDouble:elem]];
				}
			}
			
			// Store the vector in the transitory dictionary
			NSString *word= [[NSString alloc] initWithCString:wordStr encoding:NSUTF8StringEncoding];
            if ([word isEqualToString:@"</s>"])
                continue;
            
			[vectorDictionary setObject:vector forKey:word];
		}
		
	} @catch (NSException *e) {
		@throw e;
		
	} @finally {
		fclose(f);
	}
	
	return [[MLWordVectorDictionary alloc] initWithDictionary:vectorDictionary];
}

+ (MLWordVectorDictionary *) createFromGloVeFile:(NSString *)vectorFilePath {

    // Checks
    NSFileManager *fileManger= [NSFileManager defaultManager];
    if (![fileManger fileExistsAtPath:vectorFilePath])
        @throw [MLWordVectorException wordVectorExceptionWithReason:@"File does not exist"
                                                           userInfo:@{@"filePath": vectorFilePath}];
    
    // Prepare the reader
    IOLineReader *reader= [[IOLineReader alloc] initWithFilePath:vectorFilePath];

    NSUInteger vectorSize= 0;
    NSMutableDictionary *vectorDictionary= nil;
    @try {
        
        // Prepare the transitory dictionary
        vectorDictionary= [[NSMutableDictionary alloc] init];
        
        // Loop until the end of file
        do {
            
            // Read next line
            NSString *line= [reader readLine];
            if (!line)
                break;
            
            // Split the line
            NSArray *components= [[line stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]]
                                  componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
            
            // Check vector size
            if (!vectorSize) {
                vectorSize= components.count -1;
            
            } else {
                if ((components.count -1) != vectorSize)
                    @throw [MLWordVectorException wordVectorExceptionWithReason:@"Vector size mismatch"
                                                                       userInfo:@{@"filePath": vectorFilePath,
                                                                                  @"lineNumber": [NSNumber numberWithUnsignedInteger:reader.lineNumber]}];
            }
            
            // Get the word
            NSString *word= [components objectAtIndex:0];
            if ([word isEqualToString:@"<unk>"])
                continue;
            
            // Prepare the vector
            NSMutableArray *vector= [[NSMutableArray alloc] initWithCapacity:components.count -1];
            for (NSUInteger j= 1; j <= vectorSize; j++) {
                
                // Store the vector element
                double elem= [[components objectAtIndex:j] doubleValue];
                [vector addObject:[NSNumber numberWithDouble:elem]];
            }
            
            // Store the vector in the transitory dictionary
            [vectorDictionary setObject:vector forKey:word];
            
        } while (YES);
        
    } @catch (NSException *e) {
        @throw e;
        
    } @finally {
        [reader close];
    }
    
    return [[MLWordVectorDictionary alloc] initWithDictionary:vectorDictionary];
}

+ (MLWordVectorDictionary *) createFromFastTextFile:(NSString *)vectorFilePath {
    
    // Checks
    NSFileManager *fileManger= [NSFileManager defaultManager];
    if (![fileManger fileExistsAtPath:vectorFilePath])
        @throw [MLWordVectorException wordVectorExceptionWithReason:@"File does not exist"
                                                           userInfo:@{@"filePath": vectorFilePath}];
    
    // Prepare the reader
    IOLineReader *reader= [[IOLineReader alloc] initWithFilePath:vectorFilePath];
    
    NSUInteger vectorSize= 0;
    NSMutableDictionary *vectorDictionary= nil;
    @try {
        
        // Prepare the transitory dictionary
        vectorDictionary= [[NSMutableDictionary alloc] init];
        
        // Loop until the end of file
        BOOL firstLine= YES;
        do {
            
            // Read next line
            NSString *line= [reader readLine];
            if (!line)
                break;
            
            // Split the line
            NSArray *components= [[line stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]]
                                  componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
            
            if (firstLine) {
                firstLine= NO;

                // First line contains number of vectors and vector size
                vectorSize= [[components objectAtIndex:1] integerValue];
                continue;
            }
            
            // Check vector size
            if ((components.count -1) != vectorSize)
                @throw [MLWordVectorException wordVectorExceptionWithReason:@"Vector size mismatch"
                                                                   userInfo:@{@"filePath": vectorFilePath,
                                                                              @"lineNumber": [NSNumber numberWithUnsignedInteger:reader.lineNumber]}];
            
            // Get the word and convert to lowercase,
            // since fastText is case sensitive
            NSString *word= [[components objectAtIndex:0] lowercaseString];
            if ([word isEqualToString:@"</s>"])
                continue;
            
            // Prepare the vector
            NSMutableArray *vector= [[NSMutableArray alloc] initWithCapacity:components.count -1];
            for (NSUInteger j= 1; j <= vectorSize; j++) {
                
                // Store the vector element
                double elem= [[components objectAtIndex:j] doubleValue];
                [vector addObject:[NSNumber numberWithDouble:elem]];
            }
            
            // Store the vector in the transitory dictionary but
            // avoid overwriting duplicates, since we want to keep
            // the most frequent word in case of omographies with
            // different cases (e.g. "us" vs "US")
            if (![vectorDictionary objectForKey:word])
                [vectorDictionary setObject:vector forKey:word];
            
        } while (YES);
        
    } @catch (NSException *e) {
        @throw e;
        
    } @finally {
        [reader close];
    }
    
    return [[MLWordVectorDictionary alloc] initWithDictionary:vectorDictionary];
}

- (instancetype) initWithDictionary:(NSDictionary *)vectorDictionary {
	if ((self = [super init])) {
	
		// Initialization
		_vectors= [[NSMutableDictionary alloc] initWithCapacity:vectorDictionary.count];
		
		_vectorSize= 0;
		for (NSObject *key in [vectorDictionary allKeys]) {
			if (![key isKindOfClass:[NSString class]])
				@throw [MLWordVectorException wordVectorExceptionWithReason:@"Dictionary keys must be strings"
																   userInfo:@{@"dictionaryKey": key}];
			
			NSObject *vectorObj= [vectorDictionary objectForKey:key];
			if (![vectorObj isKindOfClass:[NSArray class]])
				@throw [MLWordVectorException wordVectorExceptionWithReason:@"Dictionary values must be arrays of numbers"
																   userInfo:@{@"dictionaryValue": vectorObj}];
			
            NSString *word= (NSString *) key;
			NSArray *vectorArray= (NSArray *) vectorObj;
			
			if (!_vectorSize) {
				_vectorSize= vectorArray.count;
				
				if (!_vectorSize)
					@throw [MLWordVectorException wordVectorExceptionWithReason:@"Vectors must contain at least a number"
																	   userInfo:@{@"vectorSize": [NSNumber numberWithUnsignedInteger:_vectorSize],
																				  @"word": word}];
			
			} else {
				if (vectorArray.count != _vectorSize)
					@throw [MLWordVectorException wordVectorExceptionWithReason:@"Vector size mismatch"
																	   userInfo:@{@"expectedVectorSize": [NSNumber numberWithUnsignedInteger:_vectorSize],
																				  @"actualVectorSize": [NSNumber numberWithUnsignedInteger:vectorArray.count],
																				  @"word": word}];
			}
			
			// Creation of vector
			MLReal *vector= MLAllocRealBuffer(_vectorSize);
			
			// Fill vector from array
			NSUInteger i= 0;
			for (NSObject *elemObj in vectorArray) {
				if (![elemObj isKindOfClass:[NSNumber class]])
					@throw [MLWordVectorException wordVectorExceptionWithReason:@"Dictionary values must be arrays of numbers"
																	   userInfo:@{@"vectorElement": elemObj,
																				  @"word": word,
																				  @"index": [NSNumber numberWithUnsignedInteger:i]}];

				NSNumber *elem= (NSNumber *) elemObj;
				vector[i]= (MLReal) [elem doubleValue];
				i++;
			}
			
			// Normalization of vector
			MLReal normL2= 0.0;
			ML_SVESQ(vector, 1, &normL2, _vectorSize);
			normL2= ML_SQRT(normL2);
			
			ML_VSDIV(vector, 1, &normL2, vector, 1, _vectorSize);
			
			// Creation of vector wrapper
            MLWordVector *wordVector= [[MLWordVector alloc] initWithVector:vector
                                                                      size:_vectorSize
                                                       freeVectorOnDealloc:YES];

            NSString *lowercaseWord= [word lowercaseString];
			[_vectors setObject:wordVector forKey:lowercaseWord];
		}
        
        _wordCount= _vectors.count;
	}
	
	return self;
}


#pragma mark -
#pragma mark Map lookup

- (BOOL) containsWord:(NSString *)word {
	NSString *lowercaseWord= [word lowercaseString];
	
	return ([_vectors objectForKey:lowercaseWord] != nil);
}

- (MLWordVector *) vectorForWord:(NSString *)word {
	NSString *lowercaseWord= [word lowercaseString];
	
	return [_vectors objectForKey:lowercaseWord];
}

- (NSString *) mostSimilarWordToVector:(MLWordVector *)vector {
	NSString *mostSimilarWord= nil;
	MLReal bestSimilarity= -INFINITY;
	
	// We use a sequential scan for now, slow but secure;
	// an improved search (based on clusters) will follow
	for (NSString *word in [_vectors allKeys]) {
		MLWordVector *otherVector= [_vectors objectForKey:word];
        
		MLReal similarity= [vector similarityToVector:otherVector];
		if (similarity > bestSimilarity) {
			bestSimilarity= similarity;
			mostSimilarWord= word;
		}
	}
	
	return mostSimilarWord;
}

- (NSString *) nearestWordToVector:(MLWordVector *)vector {
	NSString *nearestWord= nil;
	MLReal minorDistance= INFINITY;
	
	// We use a sequential scan for now, slow but secure;
	// an improved search (based on clusters) will follow
	for (NSString *word in [_vectors allKeys]) {
		MLWordVector *otherVector= [_vectors objectForKey:word];
        
		MLReal distance= [vector distanceToVector:otherVector];
		if (distance < minorDistance) {
			minorDistance= distance;
			nearestWord= word;
		}
	}
	
	return nearestWord;
}

- (NSArray *) mostSimilarWordsToVector:(MLWordVector *)vector {
	NSArray *sortedKeys= [[_vectors allKeys] sortedArrayWithOptions:NSSortConcurrent usingComparator:^NSComparisonResult(id obj1, id obj2) {
        MLWordVector *otherVector1= [self->_vectors objectForKey:obj1];
        MLWordVector *otherVector2= [self->_vectors objectForKey:obj2];
		
		MLReal similarity1= [vector similarityToVector:otherVector1];
		MLReal similarity2= [vector similarityToVector:otherVector2];
		
		if (similarity1 < similarity2)
			return NSOrderedDescending;
		else if (similarity1 > similarity2)
			return NSOrderedAscending;
		else
			return NSOrderedSame;
	}];
	
	return sortedKeys;
}

- (NSArray *) nearestWordsToVector:(MLWordVector *)vector {
    NSArray *sortedKeys= [[_vectors allKeys] sortedArrayWithOptions:NSSortConcurrent usingComparator:^NSComparisonResult(id obj1, id obj2) {
        MLWordVector *otherVector1= [self->_vectors objectForKey:obj1];
		MLWordVector *otherVector2= [self->_vectors objectForKey:obj2];
		
		MLReal distance1= [vector distanceToVector:otherVector1];
		MLReal distance2= [vector distanceToVector:otherVector2];
		
		if (distance1 < distance2)
			return NSOrderedAscending;
		else if (distance1 > distance2)
			return NSOrderedDescending;
		else
			return NSOrderedSame;
	}];
	
	return sortedKeys;
}


#pragma mark -
#pragma mark Properties

@synthesize wordCount= _wordCount;
@synthesize vectorSize= _vectorSize;


@end

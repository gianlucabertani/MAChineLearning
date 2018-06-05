//
//  MLWordVectorDictionary.m
//  MAChineLearning
//
//  Created by Gianluca Bertani on 03/06/15.
//  Copyright (c) 2015-2018 Gianluca Bertani. All rights reserved.
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

#import "MLBagOfWords.h"
#import "MLAlloc.h"

#import "IOLineReader.h"

#define WORD2VEC_MAX_WORD_LENGTH            (200)


#pragma mark -
#pragma mark MLWordVectorDictionary extension

@interface MLWordVectorDictionary () {
    NSUInteger _wordCount;
    NSUInteger _vectorSize;
    
    NSDictionary<NSString *, MLWordVector *> *_vectors;
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

    NSMutableDictionary<NSString *, MLWordVector *> *vectorDictionary= nil;
    @try {
        int result= 0;
        
        unsigned long dictionarySize= 0;
        unsigned long vectorSize= 0;

        result= fscanf(f, "%lu", &dictionarySize);
        if (result != 1)
            @throw [MLWordVectorException wordVectorExceptionWithReason:@"Error while reading the dictionary size"
                                                               userInfo:@{@"result": @(result)}];
        
        result= fscanf(f, "%lu", &vectorSize);
        if (result != 1)
            @throw [MLWordVectorException wordVectorExceptionWithReason:@"Error while reading the vector size"
                                                               userInfo:@{@"result": @(result)}];
        
        // Prepare the dictionary
        vectorDictionary= [[NSMutableDictionary alloc] initWithCapacity:dictionarySize];

        // Loop for all the words
        for (NSUInteger i= 0; i < dictionarySize; i++) {
            @autoreleasepool {
                
                // Read the word
                char wordStr[WORD2VEC_MAX_WORD_LENGTH];
                result= fscanf(f, "%s ", wordStr);
                if (result != 1)
                    @throw [MLWordVectorException wordVectorExceptionWithReason:@"Error while reading the next word"
                                                                       userInfo:@{@"result": @(result)}];
                
                // Read the vector values
                NSMutableArray<NSNumber *> *vectorValues= [[NSMutableArray alloc] initWithCapacity:vectorSize];
                for (NSUInteger j= 0; j < vectorSize; j++) {
                    
                    // Read and store the vector value
                    if (binary) {
                        float elem= 0.0;

                        result= (int) fread(&elem, sizeof(float), 1, f);
                        if (result != 1)
                            @throw [MLWordVectorException wordVectorExceptionWithReason:@"Error while reading a vector element"
                                                                               userInfo:@{@"result": @(result)}];
                        
                        [vectorValues addObject:@(elem)];
                        
                    } else {
                        double elem= 0.0;
                        
                        result= fscanf(f, "%lf", &elem);
                        if (result != 1)
                            @throw [MLWordVectorException wordVectorExceptionWithReason:@"Error while reading a vector element"
                                                                               userInfo:@{@"result": @(result)}];
                        
                        [vectorValues addObject:@(elem)];
                    }
                }
                
                // Get the word and skip the end-of-sentece word
                NSString *word= [[NSString alloc] initWithCString:wordStr encoding:NSUTF8StringEncoding];
                if ([word isEqualToString:@"</s>"])
                    continue;
                
                // Creation of vector
                MLReal *vector= MLAllocRealBuffer(vectorSize);
                
                // Fill vector values
                NSUInteger i= 0;
                for (NSNumber *value in vectorValues) {
                    vector[i]= (MLReal) value.doubleValue;
                    i++;
                }
                
                // Normalization of vector
                MLReal normL2= 0.0;
                ML_SVESQ(vector, 1, &normL2, vectorSize);
                normL2= ML_SQRT(normL2);
                
                ML_VSDIV(vector, 1, &normL2, vector, 1, vectorSize);
                
                // Creation of vector wrapper
                MLWordVector *wordVector= [[MLWordVector alloc] initWithVector:vector
                                                                          size:vectorSize
                                                           freeVectorOnDealloc:YES];
                
                // Store the vector in the dictionary but avoid
                // overwriting duplicates, since we want to keep
                // the most frequent word in case of omographies with
                // different cases (e.g. "us" vs "US")
                NSString *lowercaseWord= word.lowercaseString;
                if (!vectorDictionary[lowercaseWord])
                    vectorDictionary[lowercaseWord]= wordVector;
            }
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
    NSMutableDictionary<NSString *, MLWordVector *> *vectorDictionary= nil;
    @try {
        
        // Prepare the dictionary
        vectorDictionary= [[NSMutableDictionary alloc] init];
        
        // Loop until the end of file
        do {
            @autoreleasepool {
            
                // Read next line
                NSString *line= [reader readLine];
                if (!line)
                    break;
                
                // Split the line
                NSArray<NSString *> *components= [[line stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]]
                                                  componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
                
                // Check vector size
                if (!vectorSize) {
                    vectorSize= components.count -1;
                
                } else {
                    if ((components.count -1) != vectorSize)
                        @throw [MLWordVectorException wordVectorExceptionWithReason:@"Vector size mismatch"
                                                                           userInfo:@{@"filePath": vectorFilePath,
                                                                                      @"lineNumber": @(reader.lineNumber)}];
                }
                
                // Get the word and skip unknown words
                NSString *word= components[0];
                if ([word isEqualToString:@"<unk>"])
                    continue;
                
                // Creation of vector
                MLReal *vector= MLAllocRealBuffer(vectorSize);
                
                // Fill vector values
                for (NSUInteger j= 1; j <= vectorSize; j++) {
                    NSString *value= components[j];
                    vector[j -1]= (MLReal) value.doubleValue;
                }
                
                // Normalization of vector
                MLReal normL2= 0.0;
                ML_SVESQ(vector, 1, &normL2, vectorSize);
                normL2= ML_SQRT(normL2);
                
                ML_VSDIV(vector, 1, &normL2, vector, 1, vectorSize);
                
                // Creation of vector wrapper
                MLWordVector *wordVector= [[MLWordVector alloc] initWithVector:vector
                                                                          size:vectorSize
                                                           freeVectorOnDealloc:YES];
                
                // Store the vector in the dictionary but avoid
                // overwriting duplicates, since we want to keep
                // the most frequent word in case of omographies with
                // different cases (e.g. "us" vs "US")
                NSString *lowercaseWord= word.lowercaseString;
                if (!vectorDictionary[lowercaseWord])
                    vectorDictionary[lowercaseWord]= wordVector;
            }
            
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
    NSMutableDictionary<NSString *, MLWordVector *> *vectorDictionary= nil;
    @try {
        
        // Prepare the dictionary
        vectorDictionary= [[NSMutableDictionary alloc] init];
        
        // Loop until the end of file
        BOOL firstLine= YES;
        do {
            @autoreleasepool {
            
                // Read next line
                NSString *line= [reader readLine];
                if (!line)
                    break;
                
                // Split the line
                NSArray<NSString *> *components= [[line stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]]
                                                  componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
                
                if (firstLine) {
                    firstLine= NO;

                    // First line contains number of vectors and vector size
                    vectorSize= components[1].integerValue;
                    continue;
                }
                
                // Check vector size
                if ((components.count -1) != vectorSize)
                    @throw [MLWordVectorException wordVectorExceptionWithReason:@"Vector size mismatch"
                                                                       userInfo:@{@"filePath": vectorFilePath,
                                                                                  @"lineNumber": @(reader.lineNumber)}];
                
                // Get the word and and skip the end-of-sentence word
                NSString *word= components[0];
                if ([word isEqualToString:@"</s>"])
                    continue;
                
                // Creation of vector
                MLReal *vector= MLAllocRealBuffer(vectorSize);
                
                // Fill vector values
                for (NSUInteger j= 1; j <= vectorSize; j++) {
                    NSString *value= components[j];
                    vector[j -1]= (MLReal) value.doubleValue;
                }
                
                // Normalization of vector
                MLReal normL2= 0.0;
                ML_SVESQ(vector, 1, &normL2, vectorSize);
                normL2= ML_SQRT(normL2);
                
                ML_VSDIV(vector, 1, &normL2, vector, 1, vectorSize);
                
                // Creation of vector wrapper
                MLWordVector *wordVector= [[MLWordVector alloc] initWithVector:vector
                                                                          size:vectorSize
                                                           freeVectorOnDealloc:YES];
                
                // Store the vector in the dictionary but avoid
                // overwriting duplicates, since we want to keep
                // the most frequent word in case of omographies with
                // different cases (e.g. "us" vs "US")
                NSString *lowercaseWord= word.lowercaseString;
                if (!vectorDictionary[lowercaseWord])
                    vectorDictionary[lowercaseWord]= wordVector;
            }
            
        } while (YES);
        
    } @catch (NSException *e) {
        @throw e;
        
    } @finally {
        [reader close];
    }
    
    return [[MLWordVectorDictionary alloc] initWithDictionary:vectorDictionary];
}

- (instancetype) init {
    @throw [MLWordVectorException wordVectorExceptionWithReason:@"MLWordVectorDictionary class must be initialized properly"
                                                       userInfo:nil];
}

- (instancetype) initWithDictionary:(NSDictionary<NSString *, MLWordVector *> *)vectorDictionary {
    if ((self = [super init])) {
    
        // Initialization
        _vectors= vectorDictionary;
        _vectorSize= vectorDictionary.allValues.firstObject.size;
        _wordCount= _vectors.count;
    }
    
    return self;
}


#pragma mark -
#pragma mark Word lookup and comparison

- (BOOL) containsWord:(NSString *)word {
    NSString *lowercaseWord= word.lowercaseString;
    
    return (_vectors[lowercaseWord] != nil);
}

- (MLWordVector *) vectorForWord:(NSString *)word {
    NSString *lowercaseWord= word.lowercaseString;
    
    return _vectors[lowercaseWord];
}

- (NSString *) mostSimilarWordToVector:(MLWordVector *)vector {
    NSString *mostSimilarWord= nil;
    MLReal bestSimilarity= -INFINITY;
    
    // We use a sequential scan for now, slow but secure;
    // an improved search (based on clusters) will follow
    for (NSString *word in _vectors.allKeys) {
        MLWordVector *otherVector= _vectors[word];
        
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
    for (NSString *word in _vectors.allKeys) {
        MLWordVector *otherVector= _vectors[word];
        
        MLReal distance= [vector distanceToVector:otherVector];
        if (distance < minorDistance) {
            minorDistance= distance;
            nearestWord= word;
        }
    }
    
    return nearestWord;
}

- (NSArray<NSString *> *) mostSimilarWordsToVector:(MLWordVector *)vector {
    NSArray<NSString *> *sortedKeys= [_vectors.allKeys sortedArrayWithOptions:NSSortConcurrent usingComparator:^NSComparisonResult(id obj1, id obj2) {
        MLWordVector *otherVector1= self->_vectors[obj1];
        MLWordVector *otherVector2= self->_vectors[obj2];
        
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

- (NSArray<NSString *> *) nearestWordsToVector:(MLWordVector *)vector {
    NSArray<NSString *> *sortedKeys= [_vectors.allKeys sortedArrayWithOptions:NSSortConcurrent usingComparator:^NSComparisonResult(id obj1, id obj2) {
        MLWordVector *otherVector1= self->_vectors[obj1];
        MLWordVector *otherVector2= self->_vectors[obj2];
        
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
#pragma mark Sentence lookup and comparison

- (MLWordVector *) vectorForSentence:(NSString *)sentence {
    return [self vectorForSentence:sentence withLanguage:nil];
}

- (MLWordVector *) vectorForSentence:(NSString *)sentence withLanguage:(NSString *)languageCode {
    return [self vectorForSentence:sentence withLanguage:languageCode extractorType:MLWordExtractorTypeLinguisticTagger options:0 wordNotFound:nil];
}

- (MLWordVector *) vectorForSentence:(NSString *)sentence withLanguage:(NSString *)languageCode extractorType:(MLWordExtractorType)extractorType options:(MLWordExtractorOption)options wordNotFound:(void (^)(NSString *))wordNotFoundHandler {
    if (!languageCode) {
        
        // Guess the language
        switch (extractorType) {
            case MLWordExtractorTypeSimpleTokenizer: {
                languageCode= [MLBagOfWords guessLanguageCodeWithStopWordsForText:sentence];
                if (!languageCode)
                    @throw [MLWordVectorException wordVectorExceptionWithReason:@"Couldn't guess the sentence language using stop words, try using the linguistic tagger"
                                                                       userInfo:@{@"sentence": sentence}];
                
                break;
            }
                
            case MLWordExtractorTypeLinguisticTagger: {
                languageCode= [MLBagOfWords guessLanguageCodeWithLinguisticTaggerForText:sentence];
                if (!languageCode)
                    @throw [MLWordVectorException wordVectorExceptionWithReason:@"Couldn't guess the sentence language using the linguistic tagger"
                                                                       userInfo:@{@"sentence": sentence}];
                
                break;
            }
        }
    }
    
    // Split the sentence
    NSArray<NSString *> *words= nil;
    switch (extractorType) {
        case MLWordExtractorTypeSimpleTokenizer: {
            words= [MLBagOfWords extractWordsWithSimpleTokenizerFromText:sentence
                                                            withLanguage:languageCode
                                                        extractorOptions:options];
            
            break;
        }
            
        case MLWordExtractorTypeLinguisticTagger: {
            words= [MLBagOfWords extractWordsWithLinguisticTaggerFromText:sentence
                                                             withLanguage:languageCode
                                                         extractorOptions:options];
            
            break;
        }
    }

    // Check that we have something to compute the vector on
    if (words.count == 0)
        @throw [MLWordVectorException wordVectorExceptionWithReason:@"Sentence reduced to nothing with the specified extractor options"
                                                           userInfo:@{@"sentence": sentence}];

    // Compute the sentence vector
    MLWordVector *sentenceVector= nil;
    for (NSString *word in words) {
        @autoreleasepool {
            MLWordVector *wordVector= [self vectorForWord:word.lowercaseString];
            if (!wordVector) {
                if (wordNotFoundHandler)
                    wordNotFoundHandler(word);

                continue;
            }
            
            if (sentenceVector) {
                sentenceVector= [sentenceVector addVector:wordVector];
                
            } else
                sentenceVector= wordVector;
        }
    }
    
    // Check also that we found at least one word in the dictionary
    if (!sentenceVector)
        @throw [MLWordVectorException wordVectorExceptionWithReason:@"No words could be found in the dictionary"
                                                           userInfo:@{@"sentence": sentence}];

    // Normalization of vector
    MLReal magnitude= sentenceVector.magnitude;
    MLReal *mag1sentenceVector= MLAllocRealBuffer(sentenceVector.size);
    ML_VSDIV(sentenceVector.vector, 1, &magnitude, mag1sentenceVector, 1, sentenceVector.size);
    
    // Return the resulting vector
    return [[MLWordVector alloc] initWithVector:mag1sentenceVector size:sentenceVector.size freeVectorOnDealloc:YES];
}


#pragma mark -
#pragma mark Properties

@synthesize wordCount= _wordCount;
@synthesize vectorSize= _vectorSize;


@end

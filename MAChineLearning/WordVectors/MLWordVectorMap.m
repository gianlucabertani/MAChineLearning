//
//  MLWordVectorMap.m
//  MAChineLearning
//
//  Created by Gianluca Bertani on 03/06/15.
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

#import "MLWordVectorMap.h"
#import "MLWordVector.h"
#import "MLWordVectorException.h"
#import "MLWordDictionary.h"
#import "MLWordInfo.h"
#import "MLNeuralNetwork.h"
#import "MLNeuronLayer.h"
#import "MLNeuron.h"

#import "MLConstants.h"

#import <Accelerate/Accelerate.h>


#pragma mark -
#pragma mark MLWordVectorMap extension

@interface MLWordVectorMap () {
	NSMutableDictionary *_vectors;
}

@end


#pragma mark -
#pragma mark MLWordVectorMap implementation

@implementation MLWordVectorMap


#pragma mark -
#pragma mark Initialization

+ (MLWordVectorMap *) mapWithWord2vecNeuralNet:(MLNeuralNetwork *)net dictionary:(MLWordDictionary *)dictionary {
	return [[MLWordVectorMap alloc] initWithWord2vecNeuralNet:net dictionary:dictionary];
}

- (instancetype) initWithWord2vecNeuralNet:(MLNeuralNetwork *)net dictionary:(MLWordDictionary *)dictionary {
	if ((self = [super init])) {
		
		// Checks
		if (net.layers.count != 3)
			@throw [MLWordVectorException wordVectorExceptionWithReason:@"Neural network must have exactly 3 layers"
															   userInfo:@{@"layersCount": [NSNumber numberWithUnsignedInteger:net.layers.count]}];
		
		if ([[net.layers objectAtIndex:0] size] != dictionary.size)
			@throw [MLWordVectorException wordVectorExceptionWithReason:@"Neural network input layer must have same size as dictionary"
															   userInfo:@{@"inputLayerSize": [NSNumber numberWithUnsignedInteger:[[net.layers objectAtIndex:0] size]],
																		  @"dictionarySize": [NSNumber numberWithUnsignedInteger:dictionary.size]}];
		
		if ([[net.layers objectAtIndex:2] size] != dictionary.size)
			@throw [MLWordVectorException wordVectorExceptionWithReason:@"Neural network output layer must have same size as dictionary"
															   userInfo:@{@"outputLayerSize": [NSNumber numberWithUnsignedInteger:[[net.layers objectAtIndex:2] size]],
																		  @"dictionarySize": [NSNumber numberWithUnsignedInteger:dictionary.size]}];
		
		// Initialization
		_vectors= [[NSMutableDictionary alloc] initWithCapacity:dictionary.size];
		
		// Creation of vector map from hidden layer
		MLNeuronLayer *hiddenLayer= [net.layers objectAtIndex:1];
		NSUInteger vectorSize= hiddenLayer.size;
		
		for (MLWordInfo *wordInfo in dictionary.wordInfos) {
			
			// Creation of vector
			MLReal *vector= NULL;
			
			int err= posix_memalign((void **) &vector,
									BUFFER_MEMORY_ALIGNMENT,
									sizeof(MLReal) * vectorSize);
			if (err)
				@throw [MLWordVectorException wordVectorExceptionWithReason:@"Error while allocating buffer"
																   userInfo:@{@"buffer": @"vector",
																			  @"error": [NSNumber numberWithInt:err]}];
			
			// Fill vector from neural network hidden layer
			NSUInteger wordPos= wordInfo.position;

			int i= 0;
			for (MLNeuron *neuron in hiddenLayer.neurons) {
				vector[i]= neuron.weights[wordPos];
				i++;
			}

			// Creation of vector wrapper
			MLWordVector *wordVector= [[MLWordVector alloc] initWithVector:vector
																	  size:vectorSize
													   freeVectorOnDealloc:YES];
			
			NSString *word= [wordInfo.word lowercaseString];
			[_vectors setObject:wordVector forKey:word];
		}
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
	NSArray *sortedKeys= [[_vectors allKeys] sortedArrayUsingComparator:^NSComparisonResult(id obj1, id obj2) {
		MLWordVector *otherVector1= [_vectors objectForKey:obj1];
		MLWordVector *otherVector2= [_vectors objectForKey:obj2];
		
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
	NSArray *sortedKeys= [[_vectors allKeys] sortedArrayUsingComparator:^NSComparisonResult(id obj1, id obj2) {
		MLWordVector *otherVector1= [_vectors objectForKey:obj1];
		MLWordVector *otherVector2= [_vectors objectForKey:obj2];
		
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


@end

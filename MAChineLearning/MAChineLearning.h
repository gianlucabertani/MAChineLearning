//
//  MAChineLearning.h
//  MAChineLearning
//
//  Created by Gianluca Bertani on 12/04/15.
//  Copyright (c) 2015 Gianluca Bertani. All rights reserved.
//

#import <Foundation/Foundation.h>

//! Project version number for MAChineLearning.
FOUNDATION_EXPORT double MAChineLearningVersionNumber;

//! Project version string for MAChineLearning.
FOUNDATION_EXPORT const unsigned char MAChineLearningVersionString[];

#import <MAChineLearning/MLReal.h>
#import <MAChineLearning/MLAlloc.h>
#import <MAChineLearning/MLParallel.h>
#import <MAChineLearning/MLNeuralNetwork.h>
#import <MAChineLearning/MLNeuralNetwork.h>
#import <MAChineLearning/MLNeuralNetworkStatus.h>
#import <MAChineLearning/MLActivationFunctionType.h>
#import <MAChineLearning/MLBackPropagationType.h>
#import <MAChineLearning/MLCostFunctionType.h>
#import <MAChineLearning/MLLayer.h>
#import <MAChineLearning/MLInputLayer.h>
#import <MAChineLearning/MLNeuronLayer.h>
#import <MAChineLearning/MLNeuron.h>
#import <MAChineLearning/MLBiasNeuron.h>
#import <MAChineLearning/MLNeuralNetworkException.h>
#import <MAChineLearning/MLBagOfWords.h>
#import <MAChineLearning/MLBagOfWordsException.h>
#import <MAChineLearning/MLWordExtractorType.h>
#import <MAChineLearning/MLWordExtractorOption.h>
#import <MAChineLearning/MLFeatureNormalizationType.h>
#import <MAChineLearning/MLWordDictionary.h>
#import <MAChineLearning/MLMutableWordDictionary.h>
#import <MAChineLearning/MLWordInfo.h>
#import <MAChineLearning/MLWordVectorMap.h>
#import <MAChineLearning/MLWordVector.h>
#import <MAChineLearning/MLWordVectorException.h>
#import <MAChineLearning/MLRandom.h>
#import <MAChineLearning/IOLineReader.h>
#import <MAChineLearning/IOLineWriter.h>

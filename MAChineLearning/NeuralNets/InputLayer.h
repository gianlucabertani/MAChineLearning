//
//  InputLayer.h
//  MAChineLearning
//
//  Created by Gianluca Bertani on 01/03/15.
//  Copyright (c) 2015 Gianluca Bertani. All rights reserved.
//

#import <Foundation/Foundation.h>

#import "Real.h"

#import "Layer.h"


@interface InputLayer : Layer


#pragma mark -
#pragma mark Initialization

- (id) initWithIndex:(int)index size:(int)size;


#pragma mark -
#pragma mark Properties

@property (nonatomic, readonly) REAL *inputBuffer;


@end

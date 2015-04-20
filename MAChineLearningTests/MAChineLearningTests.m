//
//  MAChineLearningTests.m
//  MAChineLearningTests
//
//  Created by Gianluca Bertani on 12/04/15.
//  Copyright (c) 2015 Flying Dolphin Studio. All rights reserved.
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
#import <XCTest/XCTest.h>
#import <MAChineLearning/MAChineLearning.h>

#define NAND_TEST_TRAIN_CYCLES                           (3)
#define NAND_TEST_LEARNING_RATE                          (0.1)

#define BACKPROPAGATION_TEST_TRAIN_CYCLES             (3000)
#define BACKPROPAGATION_TEST_LEARNING_RATE               (0.5)
#define BACKPROPAGATION_TEST_VERIFICATION_CYCLES        (50)

#define LOAD_SAVE_TEST_TRAIN_CYCLES                   (2000)
#define LOAD_SAVE_TEST_LEARNING_RATE                     (0.8)

#define POW_2_52                          (4503599627370496.0)


#pragma mark -
#pragma mark MAChineLearningTests declaration

@interface MAChineLearningTests : XCTestCase
@end


#pragma mark -
#pragma mark MAChineLearningTests implementation

@implementation MAChineLearningTests


#pragma mark -
#pragma mark Setup and tear down

- (void)setUp {
    [super setUp];
}

- (void)tearDown {
    [super tearDown];
}


#pragma mark -
#pragma mark Tests

- (void) testNAND {
	@try {
		NeuralNetwork *net= [NeuralNetwork createNetworkWithLayerSizes:@[@3, @1] outputFunctionType:ActivationFunctionTypeStep];
		
		NSDate *begin= [NSDate date];
		
		for (int i= 0; i < NAND_TEST_TRAIN_CYCLES; i++) {
			
			// First set
			net.inputBuffer[0]= 1.0;
			net.inputBuffer[1]= 0.0;
			net.inputBuffer[2]= 0.0;
			net.expectedOutputBuffer[0]= 1.0;
			
			[net feedForward];
			[net backPropagateWithLearningRate:NAND_TEST_LEARNING_RATE];
			[net updateWeights];
			
			// Second set
			net.inputBuffer[0]= 1.0;
			net.inputBuffer[1]= 0.0;
			net.inputBuffer[2]= 1.0;
			net.expectedOutputBuffer[0]= 1.0;
			
			[net feedForward];
			[net backPropagateWithLearningRate:NAND_TEST_LEARNING_RATE];
			[net updateWeights];
			
			// Third set
			net.inputBuffer[0]= 1.0;
			net.inputBuffer[1]= 1.0;
			net.inputBuffer[2]= 0.0;
			net.expectedOutputBuffer[0]= 1.0;
			
			[net feedForward];
			[net backPropagateWithLearningRate:NAND_TEST_LEARNING_RATE];
			[net updateWeights];
			
			// Fourth set
			net.inputBuffer[0]= 1.0;
			net.inputBuffer[1]= 1.0;
			net.inputBuffer[2]= 1.0;
			net.expectedOutputBuffer[0]= 0.0;
			
			[net feedForward];
			[net backPropagateWithLearningRate:NAND_TEST_LEARNING_RATE];
			[net updateWeights];
			
			/* Uncomment to dump network status
			 
			 // Dump network status
			 NeuronLayer *layer= [net.layers objectAtIndex:1];
			 Neuron *neuron= [layer.neurons objectAtIndex:0];
			 NSLog(@"testNAND: bias: %.2f, weight 1: %.2f, weight 2: %.2f, weight 3: %.2f", neuron.bias, neuron.weights[0], neuron.weights[1], neuron.weights[2]);
			 
			 */
		}
		
		NSTimeInterval elapsed= [[NSDate date] timeIntervalSinceDate:begin];
		NSLog(@"testNAND: average training time: %.2f µs per cycle", (elapsed * 1000000.0) / (4.0 * ((double) NAND_TEST_TRAIN_CYCLES)));
		
		// First test
		net.inputBuffer[0]= 1.0;
		net.inputBuffer[1]= 0.0;
		net.inputBuffer[2]= 0.0;
		
		[net feedForward];
		
		XCTAssertEqualWithAccuracy(net.outputBuffer[0], 1.0, 0.1);
		
		// Second test
		net.inputBuffer[0]= 1.0;
		net.inputBuffer[1]= 0.0;
		net.inputBuffer[2]= 1.0;
		
		[net feedForward];
		
		XCTAssertEqualWithAccuracy(net.outputBuffer[0], 1.0, 0.1);
		
		// Third test
		net.inputBuffer[0]= 1.0;
		net.inputBuffer[1]= 1.0;
		net.inputBuffer[2]= 0.0;
		
		[net feedForward];
		
		XCTAssertEqualWithAccuracy(net.outputBuffer[0], 1.0, 0.1);
		
		// Fourth test
		net.inputBuffer[0]= 1.0;
		net.inputBuffer[1]= 1.0;
		net.inputBuffer[2]= 1.0;
		
		[net feedForward];
		
		XCTAssertEqualWithAccuracy(net.outputBuffer[0], 0.0, 0.1);
		
	} @catch (NSException *e) {
		XCTFail(@"Exception caught while testing: %@, reason: '%@', user info: %@", e.name, e.reason, e.userInfo);
	}
}

- (void) testBackpropagation {
	@try {
		NeuralNetwork *net= [NeuralNetwork createNetworkWithLayerSizes:@[@2, @2, @1] outputFunctionType:ActivationFunctionTypeLinear];
		
		NeuronLayer *layer1= [net.layers objectAtIndex:1];
		Neuron *neuron11= [layer1.neurons objectAtIndex:0];
		Neuron *neuron12= [layer1.neurons objectAtIndex:1];
		
		NeuronLayer *layer2= [net.layers objectAtIndex:2];
		Neuron *neuron2= [layer2.neurons objectAtIndex:0];
		
		NSDate *begin= [NSDate date];
		
		for (int i= 1; i <= BACKPROPAGATION_TEST_TRAIN_CYCLES + BACKPROPAGATION_TEST_VERIFICATION_CYCLES; i++) {
			double sum= [MAChineLearningTests randomDoubleWithMin:0.10 max:0.90];
			
			net.inputBuffer[0]= 3.0 * sum / 2.0;
			net.inputBuffer[1]= sum / 3.0;
			
			[net feedForward];
			
			double computedOutput= net.outputBuffer[0];
			net.expectedOutputBuffer[0] = sum;
			double delta= ABS(sum - computedOutput);
			
			if (i <= BACKPROPAGATION_TEST_TRAIN_CYCLES) {
				[net backPropagateWithLearningRate:BACKPROPAGATION_TEST_LEARNING_RATE];
				[net updateWeights];
				
				/* Uncomment to dump network status
				 
				 // Dump network status
				 NSLog(@"testBackpropagation: training cycle %d, expected: %.2f, computed: %.2f, delta: %.2f, status:\n" \
				       @"\t|B:%.2f W01:%.2f W02:%.2f|\n" \
				       @"\t                           x |B:%.2f W21:%.2f W22:%.2f|\n" \
				       @"\t|B:%.2f W01:%.2f W02:%.2f|\n\n",
				       i, sum, computedOutput, delta,
				       neuron11.bias, neuron11.weights[0], neuron11.weights[1],
				       neuron2.bias, neuron2.weights[0], neuron2.weights[1],
				       neuron12.bias, neuron12.weights[0], neuron12.weights[1]);
				 
				 */
				
			} else {
				
				// Check accuracy in the last cycles
				XCTAssertEqualWithAccuracy(delta, 0.00, 0.05);
			}
		}
		
		NSTimeInterval elapsed= [[NSDate date] timeIntervalSinceDate:begin];
		NSLog(@"testBackpropagation: average training/verification time: %.2f µs per cycle", (elapsed * 1000000.0) / ((double) BACKPROPAGATION_TEST_TRAIN_CYCLES + BACKPROPAGATION_TEST_VERIFICATION_CYCLES));
		
		// Check final weights
		XCTAssertEqualWithAccuracy(neuron11.bias, -0.35, 0.05);
		XCTAssertEqualWithAccuracy(neuron11.weights[0], 1.15, 0.05);
		XCTAssertEqualWithAccuracy(neuron11.weights[1], 0.26, 0.05);
		
		XCTAssertEqualWithAccuracy(neuron12.bias, -0.35, 0.05);
		XCTAssertEqualWithAccuracy(neuron12.weights[0], 1.15, 0.05);
		XCTAssertEqualWithAccuracy(neuron12.weights[1], 0.26, 0.05);
		
		XCTAssertEqualWithAccuracy(neuron2.bias, -1.05, 0.05);
		XCTAssertEqualWithAccuracy(neuron2.weights[0], 1.20, 0.05);
		XCTAssertEqualWithAccuracy(neuron2.weights[1], 1.20, 0.05);
		
	} @catch (NSException *e) {
		XCTFail(@"Exception caught while testing: %@, reason: '%@', user info: %@", e.name, e.reason, e.userInfo);
	}
}

- (void) testLoadSave {
	@try {
		NeuralNetwork *net= [NeuralNetwork createNetworkWithLayerSizes:@[@3, @2, @1] outputFunctionType:ActivationFunctionTypeLogistic];
		
		NeuronLayer *layer1= [net.layers objectAtIndex:1];
		Neuron *neuron11= [layer1.neurons objectAtIndex:0];
		Neuron *neuron12= [layer1.neurons objectAtIndex:1];
		
		NeuronLayer *layer2= [net.layers objectAtIndex:2];
		Neuron *neuron2= [layer2.neurons objectAtIndex:0];
		
		NSDate *begin= [NSDate date];
		
		// Train the network with random numbers
		for (int i= 1; i <= LOAD_SAVE_TEST_TRAIN_CYCLES; i++) {
			net.inputBuffer[0]= [MAChineLearningTests randomDoubleWithMin:0.0 max:5.0];
			net.inputBuffer[1]= [MAChineLearningTests randomDoubleWithMin:-2.5 max:2.5];
			net.inputBuffer[2]= [MAChineLearningTests randomDoubleWithMin:-5.0 max:0.0];
			net.expectedOutputBuffer[0]= [MAChineLearningTests randomDoubleWithMin:0.5 max:1.0];
			
			[net feedForward];
			[net backPropagateWithLearningRate:LOAD_SAVE_TEST_LEARNING_RATE];
			[net updateWeights];
		}
		
		/* Uncomment to dump network status at end of training
		 
		 // Dump network status
		 NSLog(@"testLoadSave: reference network status after %d training cycles:\n" \
		       @"\t|B:%.2f W01:%.2f W02:%.2f|\n" \
		       @"\t                           x |B:%.2f W21:%.2f W22:%.2f|\n" \
		       @"\t|B:%.2f W01:%.2f W02:%.2f|\n\n",
		       LOAD_SAVE_TEST_TRAIN_CYCLES,
		       neuron11.bias, neuron11.weights[0], neuron11.weights[1],
		       neuron2.bias, neuron2.weights[0], neuron2.weights[1],
		       neuron12.bias, neuron12.weights[0], neuron12.weights[1]);
		 
		 */
		
		NSTimeInterval elapsed= [[NSDate date] timeIntervalSinceDate:begin];
		NSLog(@"testLoadSave: average training time: %.2f µs per cycle", (elapsed * 1000000.0) / ((double) LOAD_SAVE_TEST_TRAIN_CYCLES));
		
		// Run a simple compute and save inputs and output
		double input0= [MAChineLearningTests randomDoubleWithMin:0.0 max:5.0];
		double input1= [MAChineLearningTests randomDoubleWithMin:-2.5 max:2.5];
		double input2= [MAChineLearningTests randomDoubleWithMin:-5.0 max:0.0];
		
		net.inputBuffer[0]= input0;
		net.inputBuffer[1]= input1;
		net.inputBuffer[2]= input2;
		
		[net feedForward];
		
		double output= net.outputBuffer[0];
		
		// Save the config and recreate the network
		NSDictionary *config= [net saveConfigurationToDictionary];
		NeuralNetwork *net2= [NeuralNetwork createNetworkFromConfigurationDictionary:config];
		
		NeuronLayer *layer1_2= [net2.layers objectAtIndex:1];
		Neuron *neuron11_2= [layer1_2.neurons objectAtIndex:0];
		Neuron *neuron12_2= [layer1_2.neurons objectAtIndex:1];
		
		NeuronLayer *layer2_2= [net2.layers objectAtIndex:2];
		Neuron *neuron2_2= [layer2_2.neurons objectAtIndex:0];
		
		/* Uncomment to dump network status after recreation
		 
		 // Dump network status
		 NSLog(@"testLoadSave: recreated network status:\n" \
			   @"\t|B:%.2f W01:%.2f W02:%.2f|\n" \
			   @"\t                           x |B:%.2f W21:%.2f W22:%.2f|\n" \
			   @"\t|B:%.2f W01:%.2f W02:%.2f|\n\n",
			   neuron11_2.bias, neuron11_2.weights[0], neuron11_2.weights[1],
			   neuron2_2.bias, neuron2_2.weights[0], neuron2_2.weights[1],
			   neuron12_2.bias, neuron12_2.weights[0], neuron12_2.weights[1]);
		 
		 */
		
		// Check the weights are the same
		XCTAssertEqualWithAccuracy(neuron11_2.bias, neuron11.bias, 0.0000000001);
		XCTAssertEqualWithAccuracy(neuron11_2.weights[0], neuron11.weights[0], 0.0000000001);
		XCTAssertEqualWithAccuracy(neuron11_2.weights[1], neuron11.weights[1], 0.0000000001);
		
		XCTAssertEqualWithAccuracy(neuron12_2.bias, neuron12.bias, 0.0000000001);
		XCTAssertEqualWithAccuracy(neuron12_2.weights[0], neuron12.weights[0], 0.0000000001);
		XCTAssertEqualWithAccuracy(neuron12_2.weights[1], neuron12.weights[1], 0.0000000001);
		
		XCTAssertEqualWithAccuracy(neuron2_2.bias, neuron2.bias, 0.0000000001);
		XCTAssertEqualWithAccuracy(neuron2_2.weights[0], neuron2.weights[0], 0.0000000001);
		XCTAssertEqualWithAccuracy(neuron2_2.weights[1], neuron2.weights[1], 0.0000000001);
		
		// If everything has been saved correctly, submitting the same input should give the same output
		net2.inputBuffer[0]= input0;
		net2.inputBuffer[1]= input1;
		net2.inputBuffer[2]= input2;
		
		[net2 feedForward];
		
		XCTAssertEqualWithAccuracy(net2.outputBuffer[0], output, 0.0000000001);
		
	} @catch (NSException *e) {
		XCTFail(@"Exception caught while testing: %@, reason: '%@', user info: %@", e.name, e.reason, e.userInfo);
	}
}


#pragma mark -
#pragma mark Internals

+ (NSUInteger) randomIntWithMax:(NSUInteger)max {
	NSUInteger random= 0;
	SecRandomCopyBytes(kSecRandomDefault, sizeof(random), (uint8_t *) &random);
	
	return (random % max);
}

+ (double) randomDoubleWithMin:(double)min max:(double)max {
	long long random= 0;
	SecRandomCopyBytes(kSecRandomDefault, sizeof(random), (uint8_t *) &random);
	
	double rnd= ((double) (ABS(random) >> 11)) / POW_2_52;
	return min + (rnd * (max - min));
}


@end

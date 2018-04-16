//
//  NeuralNetTests.m
//  MAChineLearningTests
//
//  Created by Gianluca Bertani on 12/04/15.
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

#if TARGET_OS_MAC
#import <Cocoa/Cocoa.h>
#else // !TARGET_OS_MAC
#import <Foundation/Foundation.h>
#endif // TARGET_OS_MAC

#import <XCTest/XCTest.h>
#import <MAChineLearning/MAChineLearning.h>

#define NAND_TEST_TRAIN_CYCLES                          (10)
#define NAND_TEST_LEARNING_RATE                          (0.1)

#define BACKPROPAGATION_TEST_TRAIN_CYCLES              (150)
#define BACKPROPAGATION_TEST_VERIFICATION_CYCLES        (50)

#define REGRESSION_TEST_TRAINING_SET                    (50)
#define REGRESSION_TEST_TRAIN_THRESHOLD                  (0.15)

#define LOAD_SAVE_TEST_TRAIN_CYCLES                    (100)
#define LOAD_SAVE_TEST_LEARNING_RATE                     (0.1)


#pragma mark -
#pragma mark NeuralNetTests declaration

@interface NeuralNetTests : XCTestCase
@end


#pragma mark -
#pragma mark NeuralNetTests implementation

@implementation NeuralNetTests


#pragma mark -
#pragma mark Setup and tear down

- (void) setUp {
    [super setUp];
}

- (void) tearDown {
    [super tearDown];
}


#pragma mark -
#pragma mark Tests

- (void) testNAND {
	@try {
		MLNeuralNetwork *net= [[MLNeuralNetwork alloc] initWithLayerSizes:@[@3, @1]
																  useBias:NO
														 costFunctionType:MLCostFunctionTypeSquaredError
													  backPropagationType:MLBackPropagationTypeStandard
													   hiddenFunctionType:MLActivationFunctionTypeLinear
													   outputFunctionType:MLActivationFunctionTypeStep];
		
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
			 MLNeuronLayer *layer= [net.layers objectAtIndex:1];
			 MLNeuron *neuron= [layer.neurons objectAtIndex:0];
			 NSLog(@"testNAND: weight 1: %.2f, weight 2: %.2f, weight 3: %.2f", neuron.weights[0], neuron.weights[1], neuron.weights[2]);
			 
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
		XCTFail(@"Exception caught while testing: %@, reason: '%@', user info: %@\nStack trace:%@", e.name, e.reason, e.userInfo, e.callStackSymbols);
	}
}

- (void) testBackpropagation {
	@try {
		MLNeuralNetwork *net= [[MLNeuralNetwork alloc] initWithLayerSizes:@[@2, @2, @1]
																  useBias:NO
														 costFunctionType:MLCostFunctionTypeSquaredError
													  backPropagationType:MLBackPropagationTypeResilient
													   hiddenFunctionType:MLActivationFunctionTypeLinear
													   outputFunctionType:MLActivationFunctionTypeLinear];
		
		MLNeuronLayer *layer1= [net.layers objectAtIndex:1];
		MLNeuron *neuron11= [layer1.neurons objectAtIndex:0];
		MLNeuron *neuron12= [layer1.neurons objectAtIndex:1];
		
		MLNeuronLayer *layer2= [net.layers objectAtIndex:2];
		MLNeuron *neuron2= [layer2.neurons objectAtIndex:0];
		
		// Set initial weights
		neuron11.weights[0]= 0.1;
		neuron11.weights[1]= 0.2;
		
		neuron12.weights[0]= 0.3;
		neuron12.weights[1]= 0.4;
		
		neuron2.weights[0]= 0.5;
		neuron2.weights[1]= 0.6;
		
		NSDate *begin= [NSDate date];
		
		for (int i= 1; i <= BACKPROPAGATION_TEST_TRAIN_CYCLES + BACKPROPAGATION_TEST_VERIFICATION_CYCLES; i++) {
			MLReal sum= i % BACKPROPAGATION_TEST_VERIFICATION_CYCLES;
			
			net.inputBuffer[0]= 3.0 * sum / 2.0;
			net.inputBuffer[1]= sum / 3.0;
			
			[net feedForward];
			
			MLReal computedOutput= net.outputBuffer[0];
			net.expectedOutputBuffer[0] = sum;
			MLReal delta= ABS(sum - computedOutput);
			
			if (i <= BACKPROPAGATION_TEST_TRAIN_CYCLES) {
				[net backPropagate];
				[net updateWeights];
				
				/* Uncomment to dump network status
				 
				 // Dump network status
				 NSLog(@"testBackpropagation: training cycle %d, expected: %.2f, computed: %.2f, delta: %.2f, status:\n" \
				       @"\t|W01:%.2f W02:%.2f|\n" \
				       @"\t                           x |W21:%.2f W22:%.2f|\n" \
				       @"\t|W01:%.2f W02:%.2f|\n\n",
				       i, sum, computedOutput, delta,
				       neuron11.weights[0], neuron11.weights[1],
				       neuron2.weights[0], neuron2.weights[1],
				       neuron12.weights[0], neuron12.weights[1]);
				 
				 */

			} else {
				
				// Check accuracy in the last cycles
				XCTAssertLessThan(delta, 0.01);
			}
		}
		
		NSTimeInterval elapsed= [[NSDate date] timeIntervalSinceDate:begin];
		NSLog(@"testBackpropagation: average training/verification time: %.2f µs per cycle", (elapsed * 1000000.0) / ((double) BACKPROPAGATION_TEST_TRAIN_CYCLES + BACKPROPAGATION_TEST_VERIFICATION_CYCLES));
		
		// Check final weights
		XCTAssertEqualWithAccuracy(neuron11.weights[0], 0.26, 0.01);
		XCTAssertEqualWithAccuracy(neuron11.weights[1], 0.36, 0.01);
		
		XCTAssertEqualWithAccuracy(neuron12.weights[0], 0.46, 0.01);
		XCTAssertEqualWithAccuracy(neuron12.weights[1], 0.56, 0.1);
		
		XCTAssertEqualWithAccuracy(neuron2.weights[0], 0.66, 0.01);
		XCTAssertEqualWithAccuracy(neuron2.weights[1], 0.76, 0.01);
		
	} @catch (NSException *e) {
		XCTFail(@"Exception caught while testing: %@, reason: '%@', user info: %@\nStack trace:%@", e.name, e.reason, e.userInfo, e.callStackSymbols);
	}
}

- (void) testLoadSave {
	@try {
		MLNeuralNetwork *net= [[MLNeuralNetwork alloc] initWithLayerSizes:@[@3, @2, @1]
																  useBias:YES
														 costFunctionType:MLCostFunctionTypeCrossEntropy
													  backPropagationType:MLBackPropagationTypeStandard
													   hiddenFunctionType:MLActivationFunctionTypeSigmoid
													   outputFunctionType:MLActivationFunctionTypeSigmoid];
		
		[net randomizeWeights];
		
		MLNeuronLayer *layer1= [net.layers objectAtIndex:1];
		MLNeuron *neuron11= [layer1.neurons objectAtIndex:0];
		MLNeuron *neuron12= [layer1.neurons objectAtIndex:1];
		
		MLNeuronLayer *layer2= [net.layers objectAtIndex:2];
		MLNeuron *neuron2= [layer2.neurons objectAtIndex:0];
		
		NSDate *begin= [NSDate date];
		
		// Train the network to compute the average of a sequence
		// of progressive numbers
		for (int i= 1; i <= LOAD_SAVE_TEST_TRAIN_CYCLES; i++) {
			MLReal base= 1.0 / ((MLReal) i);
			net.inputBuffer[0]= base - 0.07;
			net.inputBuffer[1]= base + 0.05;
			net.inputBuffer[2]= base + 0.13;
			
			[net feedForward];
			
			net.expectedOutputBuffer[0]= (net.inputBuffer[0] + net.inputBuffer[1] + net.inputBuffer[2]) / 3.0;

			[net backPropagateWithLearningRate:LOAD_SAVE_TEST_LEARNING_RATE];
			[net updateWeights];
		}
		
		/* Uncomment to dump network status at end of training
		 
		 // Dump network status
		 NSLog(@"testLoadSave: reference network status after %d training cycles:\n" \
		       @"\t|W01:%.2f W02:%.2f|\n" \
		       @"\t                           x |W21:%.2f W22:%.2f|\n" \
		       @"\t|W01:%.2f W02:%.2f|\n\n",
		       LOAD_SAVE_TEST_TRAIN_CYCLES,
		       neuron11.weights[0], neuron11.weights[1],
		       neuron2.weights[0], neuron2.weights[1],
		       neuron12.weights[0], neuron12.weights[1]);
		 
		 */
		
		NSTimeInterval elapsed= [[NSDate date] timeIntervalSinceDate:begin];
		NSLog(@"testLoadSave: average training time: %.2f µs per cycle", (elapsed * 1000000.0) / ((double) LOAD_SAVE_TEST_TRAIN_CYCLES));
		
		// Run a simple compute and save inputs and output
		MLReal base= 1.0 / [MLRandom nextUniformRealWithMin:1.0 max:LOAD_SAVE_TEST_TRAIN_CYCLES];
		MLReal input0= base - 0.07;
		MLReal input1= base + 0.05;
		MLReal input2= base + 0.13;
		
		net.inputBuffer[0]= input0;
		net.inputBuffer[1]= input1;
		net.inputBuffer[2]= input2;
		
		[net feedForward];
		
		MLReal output= net.outputBuffer[0];
		
		// Save the config and recreate the network
		NSDictionary *config= [net saveConfigurationToDictionary];
		MLNeuralNetwork *net2= [MLNeuralNetwork createNetworkFromConfigurationDictionary:config];
		
		MLNeuronLayer *layer1_2= [net2.layers objectAtIndex:1];
		MLNeuron *neuron11_2= [layer1_2.neurons objectAtIndex:0];
		MLNeuron *neuron12_2= [layer1_2.neurons objectAtIndex:1];
		
		MLNeuronLayer *layer2_2= [net2.layers objectAtIndex:2];
		MLNeuron *neuron2_2= [layer2_2.neurons objectAtIndex:0];
		
		/* Uncomment to dump network status after recreation
		 
		 // Dump network status
		 NSLog(@"testLoadSave: recreated network status:\n" \
			   @"\t|W01:%.2f W02:%.2f|\n" \
			   @"\t                           x |W21:%.2f W22:%.2f|\n" \
			   @"\t|W01:%.2f W02:%.2f|\n\n",
			   neuron11_2.weights[0], neuron11_2.weights[1],
			   neuron2_2.weights[0], neuron2_2.weights[1],
			   neuron12_2.weights[0], neuron12_2.weights[1]);
		 
		 */
		
		// Check the weights are the same
		XCTAssertEqualWithAccuracy(neuron11_2.weights[0], neuron11.weights[0], 0.0000000001);
		XCTAssertEqualWithAccuracy(neuron11_2.weights[1], neuron11.weights[1], 0.0000000001);
		
		XCTAssertEqualWithAccuracy(neuron12_2.weights[0], neuron12.weights[0], 0.0000000001);
		XCTAssertEqualWithAccuracy(neuron12_2.weights[1], neuron12.weights[1], 0.0000000001);
		
		XCTAssertEqualWithAccuracy(neuron2_2.weights[0], neuron2.weights[0], 0.0000000001);
		XCTAssertEqualWithAccuracy(neuron2_2.weights[1], neuron2.weights[1], 0.0000000001);
		
		// If everything has been saved correctly, submitting the same input should give the same output
		net2.inputBuffer[0]= input0;
		net2.inputBuffer[1]= input1;
		net2.inputBuffer[2]= input2;
		
		[net2 feedForward];
		
		MLReal output2= net2.outputBuffer[0];
		
		// Check the results of original and restored network are the same
		XCTAssertEqualWithAccuracy(output2, output, 0.0000000001);
		
	} @catch (NSException *e) {
		XCTFail(@"Exception caught while testing: %@, reason: '%@', user info: %@\nStack trace:%@", e.name, e.reason, e.userInfo, e.callStackSymbols);
	}
}


@end

//
//  main.m
//  MNIST
//
//  Created by Gianluca Bertani on 26/02/2017.
//  Copyright © 2017-2018 Gianluca Bertani. All rights reserved.
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

#import "MNISTDataset.h"

#define TRAINING_IMAGES_FILE_NAME     (@"train-images-idx3-ubyte")
#define TRAINING_LABELS_FILE_NAME     (@"train-labels-idx1-ubyte")
#define TEST_IMAGES_FILE_NAME         (@"t10k-images-idx3-ubyte")
#define TEST_LABELS_FILE_NAME         (@"t10k-labels-idx1-ubyte")

#define TRAINING_COST_LIMIT           (0.001)
#define TRAINING_GAIN_COST_LIMIT      (0.1)


int main(int argc, const char * argv[]) {
    @autoreleasepool {
        @try {
            
            
            ////////////////////////////////////////////////////////////////
            // Training
            
            NSLog(@"Loading training datasets...");

            // Load the training dataset
            MNISTDataset *trainingImageSet= [[MNISTDataset alloc] initWithFileName:TRAINING_IMAGES_FILE_NAME];
            MNISTDataset *trainingLabelSet= [[MNISTDataset alloc] initWithFileName:TRAINING_LABELS_FILE_NAME];

            NSLog(@"Training...");
            
            // Prepare the network
            MLNeuralNetwork *net= [[MLNeuralNetwork alloc] initWithLayerSizes:@[[NSNumber numberWithUnsignedInteger:trainingImageSet.itemSize],
                                                                                @300,
                                                                                [NSNumber numberWithUnsignedInteger:trainingLabelSet.itemSize]]
                                                                      useBias:YES
                                                             costFunctionType:MLCostFunctionTypeSquaredError
                                                          backPropagationType:MLBackPropagationTypeStandard
                                                           hiddenFunctionType:MLActivationFunctionTypeSigmoid
                                                           outputFunctionType:MLActivationFunctionTypeSigmoid];
            
            [net randomizeWeights];
            
            // Training loop
            int epochs= 0;
            MLReal lastCost= 0.0;
            do {
                NSDate *begin= [NSDate date];
                
                // Run an epoch
                MLReal cost= 0.0;
                for (int i= 0; i < trainingImageSet.items; i++) {
                    
                    // Fill the input buffer
                    ML_VCLR(net.inputBuffer, 1, net.inputSize);
                    ML_VADD([trainingImageSet itemAtIndex:i], 1, net.inputBuffer, 1, net.inputBuffer, 1, net.inputSize);
                    
                    // Fill the expected output buffer
                    ML_VCLR(net.expectedOutputBuffer, 1, net.outputSize);
                    ML_VADD([trainingLabelSet itemAtIndex:i], 1, net.expectedOutputBuffer, 1, net.expectedOutputBuffer, 1, net.outputSize);
                    
                    // Run the network
                    [net feedForward];
                    [net backPropagateWithLearningRate:0.1];
                    [net updateWeights];
                    
                    // Sum the cost
                    cost += net.cost;
                    
                    // Log every 1000 samples
                    if ((i > 0) && (i % 1000 == 0)) {
                        NSDate *end= [NSDate date];
                        NSTimeInterval elapsed= [end timeIntervalSinceDate:begin];
                        NSTimeInterval elapsedPerSample= elapsed / (double) i;
                        NSTimeInterval eta= (double)(trainingImageSet.items - i) * elapsedPerSample;

                        NSLog(@"  - Trained %5d samples, %2d epochs, ETA: %6.2f secs...", i, epochs, eta);
                    }
                }
                
                cost /= (MLReal) trainingImageSet.items;
                epochs++;
                
                NSLog(@"- Trained %2d epochs, current error: %7.5f", epochs, cost);
                
                // Check termination condition
                if (cost < TRAINING_COST_LIMIT) {
                    break;
                
                } else if (epochs > 1) {
                    double costGain= (lastCost - cost) / lastCost;
                    if (costGain < TRAINING_GAIN_COST_LIMIT)
                        break;
                }
                
                lastCost= cost;
                
            } while (YES);
            
            
            ////////////////////////////////////////////////////////////////
            // Test
            
            NSLog(@"Finished training, loading test dataset...");
            
            // Load the test dataset
            MNISTDataset *testImageSet= [[MNISTDataset alloc] initWithFileName:TEST_IMAGES_FILE_NAME];
            MNISTDataset *testLabelSet= [[MNISTDataset alloc] initWithFileName:TEST_LABELS_FILE_NAME];
            
            // Test loop
            NSUInteger matches= 0;
            NSDate *begin= [NSDate date];
            for (int i= 0; i < testImageSet.items; i++) {
                
                // Fill the input buffer
                ML_VCLR(net.inputBuffer, 1, net.inputSize);
                ML_VADD([testImageSet itemAtIndex:i], 1, net.inputBuffer, 1, net.inputBuffer, 1, net.inputSize);
                
                // Run the network
                [net feedForward];
                
                // Search highest value label
                MLReal value= 0.0;
                int label= -1;
                for (int j= 0; j < net.outputSize; j++) {
                    if (net.outputBuffer[j] > value) {
                        value= net.outputBuffer[j];
                        label= j;
                    }
                }
                
                if ([testLabelSet itemAtIndex:i][label] == 1.0)
                    matches++;
                
                // Log every 1000 samples
                if ((i > 0) && (i % 1000 == 0)) {
                    NSDate *end= [NSDate date];
                    NSTimeInterval elapsed= [end timeIntervalSinceDate:begin];
                    NSTimeInterval elapsedPerSample= elapsed / (double) i;
                    NSTimeInterval eta= (double)(testImageSet.items - i) * elapsedPerSample;
                    
                    NSLog(@"- Tested %5d samples, ETA: %6.2f secs...", i, eta);
                }
            }
            
            NSLog(@"Finished testing, report: %lu/%lu matches or %.2f%%, error rate: %.2f%%",
                  matches,
                  testImageSet.items,
                  100.0 * ((double) matches) / ((double) testImageSet.items),
                  100.0 * ((double) testImageSet.items - matches) / ((double) testImageSet.items));

        } @catch (NSException *e) {
            NSLog(@"Exception caught: %@, reason: %@: user info: %@\nStack trace:\n%@", e.name, e.reason, e.userInfo, e.callStackSymbols);
        }
    }
    
    return 0;
}


//
//  main.m
//  MNIST
//
//  Created by Gianluca Bertani on 26/02/2017.
//  Copyright © 2017 Gianluca Bertani. All rights reserved.
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

#import "Dataset.h"

#define TRAIN_IMAGES_FILE_NAME        (@"train-images-idx3-ubyte")
#define TRAIN_LABELS_FILE_NAME        (@"train-labels-idx1-ubyte")
#define TEST_IMAGES_FILE_NAME         (@"t10k-images-idx3-ubyte")
#define TEST_LABELS_FILE_NAME         (@"t10k-labels-idx1-ubyte")


int main(int argc, const char * argv[]) {
    @autoreleasepool {
        @try {
            NSLog(@"Loading datasets...");

            // First of all, get the dataset
            Dataset *trainingImageSet= [[Dataset alloc] initWithFileName:TRAIN_IMAGES_FILE_NAME];
            Dataset *trainingLabelSet= [[Dataset alloc] initWithFileName:TRAIN_LABELS_FILE_NAME];
            Dataset *testImageSet= [[Dataset alloc] initWithFileName:TEST_IMAGES_FILE_NAME];
            Dataset *testLabelSet= [[Dataset alloc] initWithFileName:TEST_LABELS_FILE_NAME];
            
            // !! TODO: da completare
            
            // insert code here...
            NSLog(@"Finished!");

        } @catch (NSException *e) {
            NSLog(@"Exception caught: %@, reason: %@: user info: %@\nStack trace: %@", e.name, e.reason, e.userInfo, e.callStackSymbols);
        }
    }
    
    return 0;
}


//
//  Dataset.m
//  MAChineLearning
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

#import "Dataset.h"

#define BASE_URL                      (@"http://yann.lecun.com/exdb/mnist/")

#define TRAIN_IMAGES_FILE_NAME        (@"train-images-idx3-ubyte")
#define TRAIN_LABELS_FILE_NAME        (@"train-labels-idx1-ubyte")
#define TEST_IMAGES_FILE_NAME         (@"t10k-images-idx3-ubyte")
#define TEST_LABELS_FILE_NAME         (@"t10k-labels-idx1-ubyte")


#pragma mark -
#pragma mark Dataset extension

@interface Dataset ()


#pragma mark -
#pragma mark Internals

- (void) downloadAndExpand:(NSString *)fileName atPath:(NSString *)path;


@end


#pragma mark -
#pragma mark Dataset implementation

@implementation Dataset


#pragma mark -
#pragma mark Initialization

- (instancetype) init {
    if ((self = [super init])) {
        
        // Get the current temp dir
        NSURL *tempUrl= [[NSFileManager defaultManager] temporaryDirectory];
        NSString *tempPath= [tempUrl path];
        
        // Check if MNIST dataset is already there
        NSString *trainImagesFilePath= [tempPath stringByAppendingPathComponent:TRAIN_IMAGES_FILE_NAME];
        if (![[NSFileManager defaultManager] fileExistsAtPath:trainImagesFilePath])
            [self downloadAndExpand:TRAIN_IMAGES_FILE_NAME atPath:tempPath];
        
        NSString *trainLabelsFilePath= [tempPath stringByAppendingPathComponent:TRAIN_LABELS_FILE_NAME];
        if (![[NSFileManager defaultManager] fileExistsAtPath:trainLabelsFilePath])
            [self downloadAndExpand:TRAIN_LABELS_FILE_NAME atPath:tempPath];

        NSString *testImagesFilePath= [tempPath stringByAppendingPathComponent:TEST_IMAGES_FILE_NAME];
        if (![[NSFileManager defaultManager] fileExistsAtPath:testImagesFilePath])
            [self downloadAndExpand:TEST_IMAGES_FILE_NAME atPath:tempPath];

        NSString *testLabelsFilePath= [tempPath stringByAppendingPathComponent:TEST_LABELS_FILE_NAME];
        if (![[NSFileManager defaultManager] fileExistsAtPath:testLabelsFilePath])
            [self downloadAndExpand:TEST_LABELS_FILE_NAME atPath:tempPath];
        
        // !! TODO: da completare
    }
    
    return self;
}


#pragma mark -
#pragma mark Internals

- (void) downloadAndExpand:(NSString *)fileName atPath:(NSString *)path {
    
    // Compose the URL
    NSURL *baseUrl= [NSURL URLWithString:BASE_URL];
    NSURL *fileUrl= [[baseUrl URLByAppendingPathComponent:fileName] URLByAppendingPathExtension:@"gz"];
    
    NSLog(@"Downloading file at URL: %@...", fileUrl);
    
    // Download
    NSError *error= nil;
    NSData *data= [NSData dataWithContentsOfURL:fileUrl options:0 error:&error];
    if (!data)
        @throw [NSException exceptionWithName:@"DatasetException"
                                       reason:@"Error while downloading dataset"
                                     userInfo:@{@"error": error}];
    
    NSLog(@"Downloaded file at URL: %@, saving to disk...", fileUrl);

    // Save to disk
    NSString *filePath= [[path stringByAppendingPathComponent:fileName] stringByAppendingPathExtension:@"gz"];
    BOOL ok= [data writeToFile:filePath options:0 error:&error];
    if (!ok)
        @throw [NSException exceptionWithName:@"DatasetException"
                                       reason:@"Error while writing downloaded file to disk"
                                     userInfo:@{@"error": error}];
    
   
    NSLog(@"Saved to disk file at path: %@, expanding...", filePath);
    
    // Expand using gunzip
    NSTask *task = [[NSTask alloc] init];
    task.currentDirectoryPath= path;
    task.launchPath = @"/usr/bin/gunzip";
    task.arguments = @[[fileName stringByAppendingPathExtension:@"gz"]];
    
    [task launch];
    [task waitUntilExit];
    
    // Check we have file file
    if (![[NSFileManager defaultManager] fileExistsAtPath:[path stringByAppendingPathComponent:fileName]])
        @throw [NSException exceptionWithName:@"DatasetException"
                                       reason:@"Failed expanding file"
                                     userInfo:@{@"path": path,
                                                @"fileName": fileName}];
    
    NSLog(@"Expanded file at path : %@", filePath);
}


@end

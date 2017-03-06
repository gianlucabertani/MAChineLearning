//
//  MNISTDataset.m
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

#import "MNISTDataset.h"

#define BASE_URL                      (@"http://yann.lecun.com/exdb/mnist/")


#pragma mark -
#pragma mark Dataset extension

@interface MNISTDataset () {
    NSUInteger _items;
    NSUInteger _itemSize;
    MLReal **_itemBuffers;
}


#pragma mark -
#pragma mark Internals

- (void) downloadAndExpand:(NSString *)fileName atPath:(NSString *)path;
- (void) readImages:(NSFileHandle *)handle;
- (void) readLabels:(NSFileHandle *)handle;


@end


#pragma mark -
#pragma mark Dataset implementation

@implementation MNISTDataset


#pragma mark -
#pragma mark Initialization

- (instancetype) initWithFileName:(NSString *)fileName {
    if ((self = [super init])) {
        
        // Get the current temp dir
        NSURL *tempUrl= [[NSFileManager defaultManager] temporaryDirectory];
        NSString *tempPath= [tempUrl path];

        NSLog(@"- Checking dataset: %@...", fileName);
        
        // Check if MNIST dataset is already there
        NSString *trainImagesFilePath= [tempPath stringByAppendingPathComponent:fileName];
        if (![[NSFileManager defaultManager] fileExistsAtPath:trainImagesFilePath])
            [self downloadAndExpand:fileName atPath:tempPath];
        
        NSLog(@"- Reading dataset: %@...", fileName);
        
        NSFileHandle *handle= [NSFileHandle fileHandleForReadingAtPath:[tempPath stringByAppendingPathComponent:fileName]];
        
        NSData *magicData= [handle readDataOfLength:4];
        UInt32 magic= CFSwapInt32(* (UInt32 *) [magicData bytes]);
        switch (magic) {
            case 0x801:
                [self readLabels:handle];
                break;
                
            case 0x803:
                [self readImages:handle];
                break;
                
            default:
                @throw [NSException exceptionWithName:@"DatasetException"
                                               reason:@"Unknown magic number"
                                             userInfo:@{@"magic": [NSNumber numberWithInteger:magic]}];
        }
    }
    
    return self;
}

- (void) dealloc {
    
    // Deallocate buffers
    for (int i= 0; i < _items; i++)
        MLFreeRealBuffer(_itemBuffers[i]);
    
    MLFreeRealPointerBuffer(_itemBuffers);
}


#pragma mark -
#pragma mark Accessors

- (MLReal *) itemAtIndex:(NSUInteger)index {
    return _itemBuffers[index];
}


#pragma mark -
#pragma mark Properties

@synthesize items= _items;
@synthesize itemSize= _itemSize;


#pragma mark -
#pragma mark Internals

- (void) downloadAndExpand:(NSString *)fileName atPath:(NSString *)path {
    
    // Compose the URL
    NSURL *baseUrl= [NSURL URLWithString:BASE_URL];
    NSURL *fileUrl= [[baseUrl URLByAppendingPathComponent:fileName] URLByAppendingPathExtension:@"gz"];
    
    NSLog(@"  - Downloading file at URL: %@...", fileUrl);
    
    // Download
    NSError *error= nil;
    NSData *data= [NSData dataWithContentsOfURL:fileUrl options:0 error:&error];
    if (!data)
        @throw [NSException exceptionWithName:@"DatasetException"
                                       reason:@"Error while downloading dataset"
                                     userInfo:@{@"error": error}];
    
    NSLog(@"  - Saving to disk...");

    // Save to disk
    NSString *filePath= [[path stringByAppendingPathComponent:fileName] stringByAppendingPathExtension:@"gz"];
    BOOL ok= [data writeToFile:filePath options:0 error:&error];
    if (!ok)
        @throw [NSException exceptionWithName:@"DatasetException"
                                       reason:@"Error while writing downloaded file to disk"
                                     userInfo:@{@"error": error}];
    
   
    NSLog(@"  - Expanding file at path: %@...", filePath);
    
    // Expand using gunzip
    NSTask *task = [[NSTask alloc] init];
    task.currentDirectoryPath= path;
    task.launchPath = @"/usr/bin/gunzip";
    task.arguments = @[[fileName stringByAppendingPathExtension:@"gz"]];
    
    [task launch];
    [task waitUntilExit];
    
    // Check we have the file
    if (![[NSFileManager defaultManager] fileExistsAtPath:[path stringByAppendingPathComponent:fileName]])
        @throw [NSException exceptionWithName:@"DatasetException"
                                       reason:@"Failed expanding file"
                                     userInfo:@{@"path": path,
                                                @"fileName": fileName}];
}

- (void) readImages:(NSFileHandle *)handle {
    
    // Read sizes
    NSData *itemsData= [handle readDataOfLength:4];
    _items= CFSwapInt32(* (UInt32 *) [itemsData bytes]);
    
    NSData *rowsData= [handle readDataOfLength:4];
    NSUInteger rows= CFSwapInt32(* (UInt32 *) [rowsData bytes]);
    
    NSData *colsData= [handle readDataOfLength:4];
    NSUInteger cols= CFSwapInt32(* (UInt32 *) [colsData bytes]);
    
    _itemSize= rows * cols;
    
    // Allocate buffer
    _itemBuffers= MLAllocRealPointerBuffer(_items);
    
    // Read images
    for (int i= 0; i < _items; i++) {
        @autoreleasepool {
            NSData *imageData= [handle readDataOfLength:_itemSize];
            UInt8 *image= (UInt8 *) [imageData bytes];
            
            _itemBuffers[i]= MLAllocRealBuffer(_itemSize);

            for (int j= 0; j < _itemSize; j++)
                _itemBuffers[i][j]= ((MLReal) image[j]) / 255.0;
        }
    }
}

- (void) readLabels:(NSFileHandle *)handle {

    // Read sizes
    NSData *itemsData= [handle readDataOfLength:4];
    _items= CFSwapInt32(* (UInt32 *) [itemsData bytes]);
    
    _itemSize= 10;
    
    // Allocate buffer
    _itemBuffers= MLAllocRealPointerBuffer(_items);
    
    // Read labels
    for (int i= 0; i < _items; i++) {
        @autoreleasepool {
            NSData *labelData= [handle readDataOfLength:1];
            UInt8 label= * (UInt8 *) [labelData bytes];
            
            _itemBuffers[i]= MLAllocRealBuffer(_itemSize);
            ML_VCLR(_itemBuffers[i], 1, _itemSize);
            _itemBuffers[i][label]= 1.0;
        }
    }
}


@end

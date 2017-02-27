//
//  MLAlloc.m
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

#import "MLAlloc.h"

#define BUFFER_MEMORY_ALIGNMENT          (128)
#define ALLOC_EXCEPTION_NAME               (@"MLAllocException")


void *mlAllocBuffer(NSUInteger itemSize, NSUInteger items, NSString *errorReason) {
    void *buffer= NULL;
    
    int err= posix_memalign((void **) &buffer, BUFFER_MEMORY_ALIGNMENT, itemSize * items);
    if (err != 0)
        @throw [NSException exceptionWithName:ALLOC_EXCEPTION_NAME
                                       reason:errorReason
                                     userInfo:@{@"error": [NSNumber numberWithInt:err]}];
    
    return buffer;
}

void mlFreeBuffer(void *buffer) {
    if (!buffer)
        return;
    
    free(buffer);
}

MLReal *mlAllocRealBuffer(NSUInteger size) {
    return mlAllocBuffer(sizeof(MLReal), size, @"Error while allocating a buffer of reals");
}

void mlFreeRealBuffer(MLReal *buffer) {
    mlFreeBuffer(buffer);
}

MLReal **mlAllocRealPointerBuffer(NSUInteger size) {
    return mlAllocBuffer(sizeof(MLReal *), size, @"Error while allocating a buffer of real pointes");
}

void mlFreeRealPointerBuffer(MLReal **buffer) {
    mlFreeBuffer(buffer);
}

int *mlAllocIntBuffer(NSUInteger size) {
    return mlAllocBuffer(sizeof(int), size, @"Error while allocating a buffer of ints");
}

void mlFreeIntBuffer(int *buffer) {
    mlFreeBuffer(buffer);
}

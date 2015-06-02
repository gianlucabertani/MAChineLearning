//
//  IOLineWriter.m
//  MAChineLearning
//
//  Created by Gianluca Bertani on 10/05/15.
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

#import "IOLineWriter.h"

#define BUFFER_SIZE                  (262144)

#define LINE_WRITER_EXCEPTION_NAME   (@"IOLineWriterException")


#pragma mark -
#pragma mark LineWriter extension

@interface IOLineWriter () {
	NSFileHandle *_file;
	NSMutableString *_buffer;
	NSCondition *_lock;
}

@end


#pragma mark -
#pragma mark LineWriter implementation

@implementation IOLineWriter


#pragma mark -
#pragma mark Initialization

+ (IOLineWriter *) lineWriterWithFilePath:(NSString *)filePath append:(BOOL)append {
	IOLineWriter *writer= [[IOLineWriter alloc] initWithFilePath:filePath append:append];
	
	return writer;
}

- (instancetype) initWithFilePath:(NSString *)filePath append:(BOOL)append {
	if ((self = [super init])) {
		
		// Initialization
		NSFileManager *fileManager= [NSFileManager defaultManager];
		
		BOOL isDirectory= NO;
		BOOL fileExists= [fileManager fileExistsAtPath:filePath isDirectory:&isDirectory];
		if (fileExists && isDirectory)
			@throw [NSException exceptionWithName:LINE_WRITER_EXCEPTION_NAME
										   reason:@"File at path already exists and is a directory"
										 userInfo:@{@"filePath": filePath}];
		
		if (!fileExists)
			[fileManager createFileAtPath:filePath contents:[NSData data] attributes:nil];

		_file= [NSFileHandle fileHandleForWritingAtPath:filePath];
		if (!_file)
			@throw [NSException exceptionWithName:LINE_WRITER_EXCEPTION_NAME
										   reason:@"File at path is read-only"
										 userInfo:@{@"filePath": filePath}];
		
		if (append)
			[_file seekToEndOfFile];

		_buffer= [[NSMutableString alloc] initWithCapacity:BUFFER_SIZE];
		_lock= [[NSCondition alloc] init];

		// Prepare locals to avoid capturing self
		NSCondition __weak *lock= _lock;
		NSMutableString __weak *buffer= _buffer;
		
		// Attach asynchronous writing block
		_file.writeabilityHandler= ^(NSFileHandle *handle) {
			@try {
				[lock lock];
				
				if (buffer.length == 0) {
					
					// Wait for a signal from writing thread
					[lock wait];
					
					// Check if the writer has been closed
					if (!buffer)
						return;
				}
				
				// Write the buffer and clear it
				NSData *outputData= [buffer dataUsingEncoding:NSUTF8StringEncoding];
				[handle writeData:outputData];
				[handle synchronizeFile];
				
				[buffer setString:@""];
				
			} @finally {
				[lock unlock];
			}
		};
	}
	
	return self;
}


#pragma mark -
#pragma mark Writing

- (void) write:(NSString *)format, ... {
	
	// Variable arguments formatting
	va_list arguments;
	va_start(arguments, format);
	NSString *text= [[NSString alloc] initWithFormat:format arguments:arguments];
	va_end(arguments);
	
	// Append string and signal the writing block
	[_lock lock];
	
	[_buffer appendString:text];
	
	[_lock signal];
	[_lock unlock];
}

- (void) writeLine:(NSString *)format, ... {
	NSMutableString *line= [NSMutableString string];
	
	// Variable arguments formatting
	va_list arguments;
	va_start(arguments, format);
	NSString *text= [[NSString alloc] initWithFormat:format arguments:arguments];
	va_end(arguments);
	
	// Append new-line and write to output stream
	[line appendString:text];
	[line appendString:@"\n"];
	
	// Append string and signal the writing block
	[_lock lock];
	
	[_buffer appendString:line];
	
	[_lock signal];
	[_lock unlock];
}

- (void) writeLine {
	
	// Append linefeed and signal the writing block
	[_lock lock];
	
	[_buffer appendString:@"\n"];
	
	[_lock signal];
	[_lock unlock];
}


#pragma mark -
#pragma mark Other operations

- (void) close {
	@try {
		[_lock lock];
		
		_buffer= nil;
		
		_file.writeabilityHandler= nil;
		[_file synchronizeFile];
		[_file closeFile];

		[_lock broadcast];
		
	} @finally {
		[_lock unlock];
	}
}


@end

//
//  IOLineReader.m
//  MAChineLearning
//
//  Created by Gianluca Bertani on 07/05/15.
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

#import "IOLineReader.h"

#define BUFFER_SIZE                  (1048576)
#define CHUNK_SIZE                   (BUFFER_SIZE / 8)
#define MAX_FILL_THRESHOLD           (7 * (BUFFER_SIZE / 8))


#pragma mark -
#pragma mark LineReader extension

@interface IOLineReader () {
	NSFileHandle *_file;
	unsigned long long _fileSize;

	NSMutableString *_buffer;
	NSCondition *_lock;
	
	NSUInteger _lastLocation;
	NSUInteger _lineNumber;
	BOOL _eof;
}

@end


#pragma mark -
#pragma mark LineReader implementation

@implementation IOLineReader


#pragma mark -
#pragma mark Initialization

- (id) initWithFileHandle:(NSString *)filePath {
	if ((self = [super init])) {
		
		// Initialization
		_file= [NSFileHandle fileHandleForReadingAtPath:filePath];
		
		_buffer= [[NSMutableString alloc] initWithCapacity:BUFFER_SIZE];
		_lock= [[NSCondition alloc] init];
		
		_lastLocation= 0;
		_lineNumber= 0;
		_eof= NO;
		
		// Get file size (we have no other way to check for EOF)
		unsigned long long fileSize= ULONG_LONG_MAX;
		NSDictionary *attribures= [[NSFileManager defaultManager] attributesOfItemAtPath:filePath error:nil];
		if (attribures)
			fileSize= [attribures fileSize];
		
		// Prepare locals to avoid capturing self
		NSFileHandle __weak *file= _file;
		NSCondition __weak *lock= _lock;
		NSMutableString __weak *buffer= _buffer;
		BOOL *eof= &_eof;

		// Attach asynchronous reading block
		_file.readabilityHandler= ^(NSFileHandle *handle) {
			
			// Read next chunk
			NSData *inputData= [handle readDataOfLength:CHUNK_SIZE];
			
			@try {
				[lock lock];
				
				// Check if the reader has been closed
				if (!buffer)
					return;
				
				// Apped line to buffer
				NSString *inputString= [[NSString alloc] initWithData:inputData encoding:NSUTF8StringEncoding];
				if (inputString)
					[buffer appendString:inputString];
				
				// Check if we are at EOF
				*eof= (file.offsetInFile == fileSize);
				
				// Signal the user's thread new data is available
				[lock broadcast];
				
				if (buffer.length >= BUFFER_SIZE) {
				
					// Wait for some room in the buffer
					// before freeing the handler
					[lock wait];
				}

			} @finally {
				[lock unlock];
			}
		};
	}
	
	return self;
}


#pragma mark -
#pragma mark Reading

- (NSString *) readLine {
	@try {
		[_lock lock];

		if (!_buffer)
			@throw [NSException exceptionWithName:@"LineReaderException"
										   reason:@"Reader is closed"
										 userInfo:nil];

		NSRange newLinePos;
		do {
			
			// Look for and end-of-line
			newLinePos= [_buffer rangeOfString:@"\n" options:0 range:NSMakeRange(_lastLocation, [_buffer length] - _lastLocation)];
			if (newLinePos.location != NSNotFound)
				break;
			
			// Check if we are at end of file
			if (_eof)
				return nil;
			
			// Wait for a signal from reading thread
			[_lock wait];
			
		} while (YES);
		
		// Extract substring and update location
		NSRange substringRange= NSMakeRange(_lastLocation, newLinePos.location +1 - _lastLocation);
		NSString *line= [_buffer substringWithRange:substringRange];
		_lastLocation= newLinePos.location +1;
		
		// If we have moved past 7/8 of the initial buffer size,
		// remove the consumed part of the buffer
		if (_lastLocation > MAX_FILL_THRESHOLD) {
			[_buffer deleteCharactersInRange:NSMakeRange(0, _lastLocation)];
			_lastLocation= 0;
			
			// Signal the reading thread there's room for new data
			[_lock broadcast];
		}
		
		_lineNumber++;
		return line;
		
	} @finally {
		[_lock unlock];
	}
}


#pragma mark -
#pragma mark Other operations

- (void) close {
	@try {
		[_lock lock];

		_buffer= nil;
		
		_file.readabilityHandler= nil;
		[_file closeFile];
		
		[_lock broadcast];
		
	} @finally {
		[_lock unlock];
	}
}


#pragma mark -
#pragma mark Properties

@synthesize lineNumber= _lineNumber;
@synthesize atEOF= _eof;


@end

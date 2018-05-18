//
//  main.m
//  StopWordsGen
//
//  Created by Gianluca Bertani on 26/04/15.
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

#import <Foundation/Foundation.h>

#define MISSING_ARGUMENT                   (9)
#define OK                                 (0)


/**
 * This utility reads *.txt files from a path, specified as the only argument,
 * and producers a .h header file as output. The header file provides the
 * definition for an NSDictionary containing all the stopwords found in the txt
 * files. It is used to generated the stop words header file StopWords.h.
 *
 * Files must be written using the syntax of stop words files of the Snowball project
 * at tartatus.org. See: http://svn.tartarus.org/snowball/trunk/website/algorithms/
 */
int main(int argc, const char * argv[]) {
	@autoreleasepool {
		if (argc != 2)
			return MISSING_ARGUMENT;
		
		NSMutableDictionary<NSString *> *stopWords= [NSMutableDictionary dictionary];
		
		// Get directory list at path of first argument
		NSFileManager *manager= [NSFileManager defaultManager];
		NSString *path= [[NSString alloc] initWithCString:argv[1] encoding:NSUTF8StringEncoding];
		NSArray<NSString *> *fileNames= [manager contentsOfDirectoryAtPath:path error:nil];

		// Loop all the files
		for (NSString *fileName in fileNames) {
			
			// Skip non-txt files
			if (![fileName hasSuffix:@".txt"])
				continue;

			NSRange extRange= [fileName rangeOfString:@".txt"];
			NSString *language= [fileName substringToIndex:extRange.location];
			
			// Get file content as text
			NSString *filePath= [path stringByAppendingPathComponent:fileName];
			NSData *fileContent= [manager contentsAtPath:filePath];
			NSString *textContent= [[NSString alloc] initWithData:fileContent encoding:NSUTF8StringEncoding];

			// Loop all the lines
			NSArray<NSString *> *lines= [textContent componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
			
			NSMutableArray<NSString *> *words= [NSMutableArray array];
			for (NSString *line in lines) {
				NSArray<NSString *> *tokens= [line componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
				
				// Skip empty lines
				if (tokens.count == 0)
					continue;
				
				// Loop all the tokens
				for (NSString *token in tokens) {
					
					// Skip empty tokens
					if (token.length == 0)
						continue;
					
					// Skip line number
					if ([token intValue] > 0)
						continue;
					
					// A "|" marks the beginning of a comment,
					// no more valid tokens after this point
					if ([token hasPrefix:@"|"])
						break;
					
					[words addObject:token];
				}
			}
			
			[stopWords setObject:words forKey:language];
		}
		
		// Prepare output text
		NSMutableString *output= [NSMutableString string];
		[output appendFormat:@"#define STOP_WORDS @{ \\\n"];
		
		for (NSString *language in [stopWords allKeys]) {
			[output appendFormat:@"\t\t@\"%@\": [NSSet setWithArray:@[", language];
			
			NSArray<NSString *> *words= [stopWords objectForKey:language];
			for (int i= 0; i < words.count; i++) {
				if (i > 0)
					[output appendString:@", "];
				
				NSString *word= [words objectAtIndex:i];
				[output appendFormat:@"@\"%@\"", word];
			}
			
			[output appendString:@"]], \\\n"];
		}
		
		[output appendString:@"\t}\n"];
		
		// Write output file
		NSString *outputPath= [path stringByAppendingPathComponent:@"output.h"];
		[manager createFileAtPath:outputPath
						 contents:[output dataUsingEncoding:NSUTF8StringEncoding]
					   attributes:nil];
	}

	return OK;
}

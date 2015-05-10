# Bag of Words with MAChineLearning

Bag of Words in MAChingLearning currently supports:

- simple tokenization or linguistic tagging;
- stop words for 12 western languages;
- token exclusion and composition by tag;
- configurations for sentiment analysis and topic classification;
- boolean and L2 normalization;
- language guessing.


Following there's a quick tutorial on how to use bag of words with your neural network.


## Tutorial

### Language guessing

For language guessing you can choose between the iOS/OS X integrated linguistic tagger, with a helper function, or an alternative algorithm that counts occurrences of stop words:

```obj-c
#import <MAChineLearning/MAChineLearning.h>

// Guess the language with linguistic tagger
NSString *lang1= [BagOfWords guessLanguageCodeWithLinguisticTagger:@"If you're not failing every now and again, it's a sign you're not doing anything very innovative."];

// Guess the language with alternative algorith,
NSString *lang2= [BagOfWords guessLanguageCodeWithStopWords:@"If you're not failing every now and again, it's a sign you're not doing anything very innovative."];
```

The language is expresses as a ISO-639-1 code, such as "en" for English, "fr" for French, etc.

### Text tokenization

To tokenize a text into a bag of words you need a dictionary: a map that tells for each possible word in the text the position that represents it in the bag of words vector.

With MAChineLearning, the dictionary is built progressively as texts are tokenized, you just need to fix its maximum size from the beginning. 

```obj-c
NSMutableDictionary *dictionary= [NSMutableDictionary dictionary];
NSUInteger dictionaryMaxSize= 50000;

// Load texts into an array
NSArray *movieReviews= // ...

for (NSString *movieReview in movieReviews) { 

	// Extract the bag of words for the current text
	BagOfWords *bag= [BagOfWords bagOfWordsForSentimentAnalysisWithText:movieReview
																 textID:nil
														     dictionary:dictionary
														 dictionarySize:dictionaryMaxSize
															   language:@"en"
												   featureNormalization:FeatureNormalizationTypeNone];

	// Dump the extracted tokens
	NSLog(@"Tokens: %@", bag.tokens);

	// The actual bag of words vector is accessible in bag.outputBuffer
	for (NSString *token in bag.tokens) {
		NSNumber *pos= [dictionary objectForKey:[token lowerCaseString]];
		REAL occurrencies= net.outputBuffer[[pos intValue]];

		NSLog(@"Occurrencies for token '%@': %.0f", token, occurrencies);
	}

	// Fill the input buffer of the neural network
	// ...
}
```

Each tokenization loop adds words to the dictionary. When a new word is encountered, it is assigned a position at the end of the dictionary. Once the loop is completed, it may look something like this:

- "think": 0,
- "majority": 1,
- "people": 2,
- "seem": 3,
- etc.


### Use in combination with a neural network

In most of use cases, bag of words are submitted as input to a neural network. With MAChineLearning, you may specify that the output buffer of the bag of words is the input buffer of the neural network. This reduces memory and time consumption.


```obj-c
for (NSString *movieReview in movieReviews) { 

	// Extract the bag of words for the current text
	BagOfWords *bag= [BagOfWords bagOfWordsForSentimentAnalysisWithText:movieReview
																 textID:nil
															 dictionary:dictionary
														 dictionarySize:net.inputSize // Use network input size
															   language:@"en"
												   featureNormalization:FeatureNormalizationTypeNone
														   outputBuffer:net.inputBuffer]; // Use network input buffer

	// You may run the network immediately
	[net feedForward];

	// Evaluate the result
	// ...
}
```

### Choosing tokenization options

The BagOfWords class provides two factory methods preconfigured for sentiment analysis and topic classification, but you may want to fine tune the tokenizer to your needs.

There are two kinds of tokenizer:

- the **simple tokenizer** simply splits the text searching for white spaces and new lines;
- the **linguistic tagger** again uses the iOS/OS X integrated linguistic tagger, which provides more in-depth knowledge on each token found.

The latter is of course better, but it is much slower, and in some applications is "a bazooka to kill a fly". E.g. in sentiment analysis you usually have better results with wider dictionaries, rather than with carefully picked word combinations.

With the **simple tokenizer**, tokenization options are:

- omit stop words;
- keep all bigrams;
- keep all trigrams;
- keep emoticons and emoji.

With the **linguistic tagger**, tokenization options are:

- omit stop words;
- omit adjectives;
- omit adverbs;
- omit nouns;
- omit names;
- omit numbers;
- omit others (conjunctions, prepositions, etc.);
- keep verb+adjective combos (e.g. "is nice");
- keep adjective+noun combos (e.g. "nice movie");
- keep adverb+noun combos (e.g. "earlier reviews");
- keep noun+noun combos (e.g. "human thought");
- keep noun+verb combos (e.g. "movies are");
- keep 2-word names (e.g. "Alan Turing");
- keep 3-word names (e.g. "Arthur Conan Doyle");
- keep all bigrams;
- keep all trigrams;
- keep emoticons and emoji.

Tokenizers and options are specified using enums `WordExtractorType` and `WordExtractorOption`, respectively.

The most extended version of the BagOfWords factory methods includes parameters to specify both the tokenizer and its tokenization options:

```obj-c
BagOfWords *bag= [BagOfWords bagOfWordsWithText:text
										 textID:textID
									 dictionary:dictionary
								 dictionarySize:net.inputSize
									   language:@"en"
								  wordExtractor:WordExtractorTypeLinguisticTagger
							   extractorOptions:WordExtractorOptionOmitStopWords | WordExtractorOptionKeepNounNounCombos | WordExtractorOptionKeep2WordNames | WordExtractorOptionKeep3WordNames
						   featureNormalization:FeatureNormalizationTypeNone
								   outputBuffer:net.inputBuffer];
```

Default configurations are the following:

- for **sentiment analysis**:
  - simple tokenizer;
  - omit stop words;
  - keep emoticons and emoji;
  - keep all bigrams.
- for **topic classification**:
  - linguistic tagger;
  - omit stop words;
  - omit verbs, adjectives, adverbs, nouns, others;
  - keep adjective+noun combos;
  - keep adverb+noun combos;
  - keep noun+noun combos;
  - keep 2-word names;
  - keep 3-word names.

You may need to experiment a bit to find the correct configuration for you.

### Choosing normalization options

When no normalization is applied, ach position on the bag of words vector specifies the number of occurences of the corresponding word in the text. Normalizing the vector means keeping its values limited, to improve the neural network convergence.

The following normalizations may be applied:

- **boolean normalization**: occurrencies are not counted and each position may hold just 1 or 0;
- **L2 normalization**: each vector position is divided by the vector's Euclidean length.

Another common used normalization method, called *TF-iDF*, is missing for now, it will be added in the future.

Take for example the following sentence:

- *"I think the majority of the people seem not the get the right idea about the movie"*

Its bag of words vector could be the following (keeping stop words and adding some more words to fill up):

![Bag Of Words Norm Example](Bag%20Of%20Words%20Norm%20Example.png)

Normalization options are specified using enum `FeatureNormalizationType`.


## Examples

The library contains some unit tests that show how to use it, see [BagOfWordsTests.m](MAChineLearningTests/BagOfWordsTests.m).


## References

The following two resources provide a clear introduction to bag of words and their implementation:

* [Bag of Words Meets Bags of Popcorn - Part 1: For Beginners - Bag of Words](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words)
* [The Vector Space Model of text](http://stanford.edu/~rjweiss/public_html/IRiSS2013/text2/notebooks/tfidf.html)


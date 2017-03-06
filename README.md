# MAChineLearning

Machine Learning for the Mac


## Intro

*MAChineLearning* (pron. ˈmækʃiːn ˈlɜːnɪŋ) is framework that provides a quick and easy way to experiment Machine Learning with native code on the Mac, with some specific support for Natural Language Processing. It is written in Objective-C, but it is compatible by Swift.

Currently the framework supports:

- [Neural Networks](#Neural Networks)
- [Bag of Words](#Bag of Words)
- [Word Vectors](#Word Vectors)

Differently than many other machine learning libraries for macOS and iOS, MAChineLearning includes full training implementation for its neural networks. You don't need a separate language or another framework to train your network, you have all you need here.


### Use on iOS

While targeted at macOS, the framework should easily recompile and work on iOS too. Training the network should also work correctly on iOS.


## Neural Networks

For an introduction to neural networks, see [Artificial neural network](https://en.wikipedia.org/wiki/Artificial_neural_network) on Wikipedia.

Neural networks in MAChineLearning currently support:

- [Multilayer perceptrons](https://en.wikipedia.org/wiki/Multilayer_perceptron) of any depth (limited only by memory).
- 5 kinds of activation functions:
  - Linear.
  - [Rectified linear (a.k.a. ReLU)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)).
  - Step (0 if output is less than 0.5, 1 if greater).
  - [Sigmoid (a.k.a. logistic)](https://en.wikipedia.org/wiki/Logistic_function).
  - [TanH (a.k.a. hyperbolic tangent)](https://en.wikipedia.org/wiki/Hyperbolic_function#Hyperbolic_tangent).
- 2 kinds of cost functions:
  - Squared error.
  - [Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression).
- 2 kinds of backpropagation:
  - Standard.
  - [Resilient (a.k.a. RPROP)](https://en.wikipedia.org/wiki/Rprop).
- Training by sample or by batch.
- Load/save of the network status from/to a dictionary.
- Single/double precision (needs recompilation, default is single precision).

Internal code makes heavy use of the [Accelerate framework](https://developer.apple.com/reference/accelerate), in particular vDSP and vecLib functions. It is as fast as it can be on a CPU. On a GPU of course would be faster, but it's already pretty damn fast (20x faster than a Java equivalent).


## Tutorial

### Setting up the network

Setting up a network is a matter of two lines:

```obj-c
#import <MAChineLearning/MAChineLearning.h>

// Create a perceptron with 3 input lines and 1 output neuron
MLNeuralNetwork *net= [NeuralNetwork createNetworkWithLayerSizes:@[@3, @1]
                                              outputFunctionType:MLActivationFunctionTypeStep];

[net randomizeWeights];
```

These lines create a single layer [perceptron](https://en.wikipedia.org/wiki/Perceptron) with 3 inputs and 1 output, with step activation function, and randomize its initial weights. See the following diagram:

![3-Input Perceptron](3-Input%20Perceptron.png)

The network object exposes all you need to control it, namely:

- The input vector.
- The output vector.
- The expected output vector.
- Methods to feed forward, back propagate and update weights.
- Methods to save the status and create a new network from a saved state.

Vectors are exposed as C buffers (arrays) for performance reason. They are of type `MLReal`, which by default is a typedef of `float`. To work with double precision, you can redefine this type to `double` in the `MLReal.h` file and then recompile. Just follow the comments.


### Loading input

This is how you load your data in the input buffer:

```obj-c
// Clear the input buffer
for (int i= 0; i < net.inputBuffer; i++)
    net.inputBuffer[i]= 0.0;

// Fill appropriate buffer elements
net.inputBuffer[0]= 1.0;
net.inputBuffer[2]= 0.5;
```

You can use Accelerate framework to clear the buffer more quickly. In this case, you can make use of macros defined in `MLReal.h`, to avoid changing function names in case you later move from single to double precision:

```obj-c
// Clear the input buffer using Accelerate
ML_VCLR(net.inputBuffer, 1, net.inputSize);

// Fill appropriate buffer elements
net.inputBuffer[0]= 1.0;
net.inputBuffer[2]= 0.5;
```


### Computing the output

Once the input buffer is filled, computing the output is simple:

```obj-c
// Compute the output
[net feedForward];

// Log the output
NSLog(@"Output: %.2f", net.outputBuffer[0]);
```


### Training

If the output is not satisfactory, you can set the expected output in its specific buffer and ask the network to backpropagate the error.

```obj-c
// Clear the expected output buffer using Accelerate
ML_VCLR(net.expectedOutputBuffer, 1, net.outputSize);

// Set the expected output
net.expectedOutputBuffer[0]= 0.5;

// Backpropagate the error
[net backPropagateWithLearningRate:0.1];
```

The network automatically computes the error and applies the gradient descent algorithm to obtain new weights. The *learning rate* parameter makes learning faster (and more uncertain) for greater values, or slower (but more certain) for lower values. When using resilient backpropagation, the learning rate must be specified as 0, since the RPROP algoritm sets its learning rate automatically.

**New weights are not applied immediately**: they are stored inside the network, so that you may run multiple feed forwards and backpropagations before applying them (i.e. train by batch).

Once your training batch is complete, update weights in the following way:

```obj-c
// Update weights
[net updateWeights];
```


#### Training loop

During training, you typically feed the full sample set to the network multiple times, so that it increases its predictive capabilities. Each complete pass for the sample set is called an *epoch*. While feeding the epoch, you may update the weights after each sample, or wait until a batch of samples have been fed (e.g. 10 samples), to slightly increase the training performance.

A typical training loop is the following:

```obj-c
BOOL finished= NO;
do {

	// Load the sample set
	// ...

    MLReal error= 0.0;
	for (int i= 0; i < numberOfSamples; i++) {

	    // Clear the input buffer with vDSP
		ML_VCLR(net.inputBuffer, 1, net.inputSize);

		// Load the i-th sample
		net.inputBuffer[0]= 1.0;
		net.inputBuffer[2]= 0.5;
		// ...

        // Feed the network and compute the output
		[net feedForward];

        // Set the expected output for the i-th sample
        net.expectedOutputBuffer[0]= 0.5;
        // ...

        // Add the error (cost) for the i-th sample
        error += net.cost;

        [net backPropagateWithLearningRate:0.1];

        // Update weights
        [net updateWeights];
	}

	// Compute the average error
	error /= (MLReal) numberOfSamples;

	// Check if average error is below the expected threshold
	finished= (error < 0.0001);

} while (!finished);
```

While a training loop with **batch updates** is the following:

```obj-c
BOOL finished= NO;
do {

	// Load the sample set
	// ...

    MLReal error= 0.0;
	for (int i= 0; i < numberOfSamples; i++) {

	    // Clear the input buffer with vDSP
		ML_VCLR(net.inputBuffer, 1, net.inputSize);

		// Load the i-th sample
		net.inputBuffer[0]= 1.0;
		net.inputBuffer[2]= 0.5;
		// ...

        // Feed the network and compute the output
		[net feedForward];

        // Set the expected output for the i-th sample
        net.expectedOutputBuffer[0]= 0.5;
        // ...

        // Add the error (cost) for the i-th sample
        error += net.cost;

        [net backPropagateWithLearningRate:0.1];

        if ((i +1) % 10 == 0) {

        	// Update weights for this batch
        	[net updateWeights];
        }
    }

	// Compute the average error
	error /= (MLReal) numberOfSamples;

	// Check if average error is below the expected threshold
	finished= (error < 0.0001);

} while (!finished);
```

The network enforces the correct calling sequence by using a simple state machine. Check the following state diagram:

![Network States](Network%20States.png)

If you try a call that does not correspond to a state transition in the above diagram, the network will throw an exception.


### Examples

The framework contains some unit tests that show how to use it, see [NeuralNetTests.m](MAChineLearningTests/NeuralNetTests.m).

The first of them is the NAND logic port discussed on [Wikipedia](http://en.wikipedia.org/wiki/Perceptron#Example). The tests includes a few commented lines that, if uncommented, dump the network status after each training. Note that if you compare it with the Wikipedia example numbers will differ. This is due to the use of bias, which the Wikipedia example does not include.


#### MNIST

A full implementation of the [MNIST example](yann.lecun.com/exdb/mnist/) for handwritten digits recognition is included, see [main.m](MNIST/main.m). It downloads automatically the dataset and trains the network until it reaches a certain confidence. Expect a typical running time around 2 minutes and a resulting error rate of 2.8%.


### References

There are a lot articles out there explaining how neural networks work, but I have found these two in particular well written and clear enough to base my coding on them:

* [Machine Learning: Multi Layer Perceptrons](http://ml.informatik.uni-freiburg.de/_media/teaching/ss10/05_mlps.printer.pdf) [PDF]
* [Designing And Implementing A Neural Network Library For Handwriting Detection, Image Analysis etc.](http://www.codeproject.com/Articles/14342/Designing-And-Implementing-A-Neural-Network-Librar)

I am grateful to these people for taking the time to share their knowledge.


## Bag of Words

*Bag of Words* is a well known method to represent a text numerically, so that it can be used to train a neural network. It is based on an vector of numbers where each element represents a word in the text, and (in its simplest form) is either set to 1 o 0 if that word occurs or not in the text to be represented. While it is considered outdated since the introduction of word vectors, it can still perform well in a number of tasks.

To build a bag of words you start from a dictionary of words, with each word assigned the index of its corresponding element in the vector. Given a text, it is then split in separate words (a process called *tokenization*) and, for every word, they are looked up in the dictionary and their corresponding element on the array is set accordingly.

A number of improvements may be applied to this process, including the removal of frequently words (called *stop words*), more or less sophisticated tokenization, normalization of the bag of words vector, et.c

The Bag of Words toolkit in MAChineLearning currently supports:

- Stop words for 12 western languages (with language guessing).
- 2 different algorithms for tokenization:
  - Linguistic tagging.
  - Simple tokenization.
- 5 different kinds of normalization:
  - Boolean.
  - L1.
  - L2.
  - L1 with TF-IDF.
  - L2 with TF-IDF.
- Pre-built configurations for sentiment analysis and topic classification.


### Tutorial

#### Building Bags of Words

With MAChineLearning, the dictionary for the Bag of Words is built progressively as texts are tokenized, you just need to fix its maximum size from the beginning. While the dictionary encompasses all the Bag of Words vectors, each Bag of Words instance represents just one text.

The following examples shows how to build the Bag of Words vectors for a set of movie reviews:

```obj-c
MLMutableWordDictionary *dictionary= [MLMutableWordDictionary dictionaryWithMaxSize:5000];

// Load texts
NSArray *movieReviews= // ...

for (NSString *movieReview in movieReviews) {

	// Extract the bag of words for the current text
	MLBagOfWords *bag= [MLBagOfWords bagOfWordsForSentimentAnalysisWithText:movieReview
						  									     documentID:nil
														         dictionary:dictionary
															       language:@"en"
												       featureNormalization:FeatureNormalizationTypeNone];

	// Dump the extracted words and their occurrences
	for (NSString *word in bag.words) {
	    MLWordInfo *info= [dictionary infoForWord:word];
		MLReal occurrencies= net.outputBuffer[info.position];

		NSLog(@"Occurrences for word '%@': %.0f", word, occurrencies);
	}
}
```

Each tokenization loop adds words to the dictionary. When a new word is encountered, it is assigned a position at the end of the dictionary, unless the dictionary is already filled up. In that case the word is discarded.


#### Language guessing

When skipping stop words, the tokenization process must know the language the text is written in. For language guessing there are two utility methods available, employing either the macOS integrated linguistic tagger or an alternative algorithm that counts occurrences of stop words:

```obj-c
#import <MAChineLearning/MAChineLearning.h>

// Guess the language with linguistic tagger
NSString *lang1= [MLBagOfWords guessLanguageCodeWithLinguisticTaggerForText:@"If you're not failing every now and again, it's a sign you're not doing anything very innovative."];

// Guess the language with stop words
NSString *lang2= [MLBagOfWords guessLanguageCodeWithStopWordsForText:@"If you're not failing every now and again, it's a sign you're not doing anything very innovative."];
```

The language is expresses as a *ISO-639-1 code*, such as "en" for English, "fr" for French, etc.


#### Using Bag of Words with a neural network

In most of use cases, Bag of Words vectors are submitted as input to a neural network. With MAChineLearning, you may specify that the output buffer of the Bag of Words is the input buffer of the neural network. This reduces memory and time consumption.

```obj-c
for (NSString *movieReview in movieReviews) {

	// Extract the bag of words for the current text
	MLBagOfWords *bag= [MLBagOfWords bagOfWordsForSentimentAnalysisWithText:movieReview
						  									     documentID:nil
															     dictionary:dictionary
															       language:@"en"
												       featureNormalization:MLFeatureNormalizationTypeNone
														       outputBuffer:net.inputBuffer]; // Use network input buffer

	// You may run the network immediately
	[net feedForward];

	// Evaluate the result
	// ...
}
```


### Choosing tokenization options

The MLBagOfWords class provides two factory methods preconfigured for sentiment analysis and topic classification, but you may want to fine tune the tokenizer to your needs.

There are 2 kinds of tokenizer:

- The **simple tokenizer** splits the text by white spaces and new lines.
- The **linguistic tagger** uses the iOS/OS X integrated linguistic tagger, which provides more in-depth knowledge on each token found.

The latter is of course better, but it is also much slower.

With the **simple tokenizer**, tokenization options are:

- Omit stop words.
- Keep all bigrams.
- Keep all trigrams.
- Keep emoticons and emoji.

With the **linguistic tagger**, tokenization options are:

- Omit stop words.
- Omit adjectives.
- Omit adverbs.
- Omit nouns.
- Omit names.
- Omit numbers.
- Omit others (conjunctions, prepositions, etc.).
- Keep verb+adjective combos (e.g. "is nice").
- Keep adjective+noun combos (e.g. "nice movie").
- Keep adverb+noun combos (e.g. "earlier reviews").
- Keep noun+noun combos (e.g. "human thought").
- Keep noun+verb combos (e.g. "movies are").
- Keep 2-word names (e.g. "Alan Turing").
- Keep 3-word names (e.g. "Arthur Conan Doyle").
- Keep all bigrams.
- Keep all trigrams.
- Keep emoticons and emoji.

Tokenizers and their options are specified using enums `MLWordExtractorType` and `MLWordExtractorOption`, respectively.

The most extended version of the MLBagOfWords factory method includes parameters to specify both the tokenizer and its tokenization options:

```obj-c
MLBagOfWords *bag= [MLBagOfWords bagOfWordsWithText:text
									     documentID:documentID
									     dictionary:dictionary
							        buildDictionary:YES
									       language:@"en"
								      wordExtractor:WordExtractorTypeLinguisticTagger
							       extractorOptions:WordExtractorOptionOmitStopWords | WordExtractorOptionKeepNounNounCombos | WordExtractorOptionKeep2WordNames | WordExtractorOptionKeep3WordNames
						       featureNormalization:FeatureNormalizationTypeNone
								       outputBuffer:net.inputBuffer];
```

Default configurations are the following:

- For **sentiment analysis**:
  - Simple tokenizer.
  - Omit stop words.
  - Keep emoticons and emoji.
  - Keep all bigrams.
- For **topic classification**:
  - Linguistic tagger.
  - Omit stop words.
  - Omit verbs, adjectives, adverbs, nouns, others.
  - Keep adjective+noun combos.
  - Keep adverb+noun combos.
  - Keep noun+noun combos.
  - Keep 2-word names.
  - Keep 3-word names.

You may need to experiment a bit to find the correct configuration for your task.


### Choosing normalization options

By applying a normalization to the Bag of Words vector, you can keep its values limited, improving chances of neural network convergence.

The following normalizations may be applied:

- **No normalization**: each position on the Bag of Words vector specifies the number of occurences of the corresponding word in the text.
- **Boolean normalization**: occurrencies are not counted and each position may hold just 1 (if the word occurs at least once) or 0 (if it does not occur).
- **L1 normalization**: each vector position is divided by the vector's [L1 norm](http://mathworld.wolfram.com/L1-Norm.html). The resulting vector length is always 1.
- **L2 normalization**: each vector position is divided by the vector's [L2 (Euclidean) norm](http://mathworld.wolfram.com/L2-Norm.html). The resulting vector length is always 1.
- **TF-iDF with L1 normalization**: word occurrences are scaled inversely to their frequency in the entire corpus of texts, according to the [TF-iDF](https://en.wikipedia.org/wiki/Tf–idf) algorithm. This means that words that occur frequently have less weight than words that occur rarely. Each vector position is then divided by the vector's [L1 length](http://mathworld.wolfram.com/L1-Norm.html). The resulting vector length is always 1.
- **TF-iDF with L2 normalization**: word occurrences are scaled inversely to their frequency in the entire corpus of texts, according to the [TF-iDF](https://en.wikipedia.org/wiki/Tf–idf) algorithm. This means that words that occur frequently have less weight than words that occur rarely. Each vector position is then divided by the vector's [L2 (Euclidean) length](http://mathworld.wolfram.com/L2-Norm.html). The resulting vector length is always 1.

Normalization options are specified using enum `MLFeatureNormalizationType`.

Given the following sample sentence:

- *"I think the majority of the people seem not the get the right idea about the movie"*

Its Bag of Words vector for no normalization, boolean normalization and L2 normalization is the following:

![Bag Of Words Norm Example](Bag%20Of%20Words%20Norm%20Example.png)


### Examples

The framework contains some unit tests that show how to use it, see [BagOfWordsTests.m](MAChineLearningTests/BagOfWordsTests.m).


### References

The following two resources provide a clear introduction to bag of words and their implementation:

* [Bag of Words Meets Bags of Popcorn - Part 1: Bag of Words](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words)
* [The Vector Space Model of text](http://stanford.edu/~rjweiss/public_html/IRiSS2013/text2/notebooks/tfidf.html)


## Word Vectors

*Word vectors*, a.k.a. *word embeddings*, are a way to represent words using high-dimensional vectors that embed their semantic relationships. They have been introduced in 2013 with the [Word2vec model](https://en.wikipedia.org/wiki/Word2vec), and are currently considered the state-of-the-art for Natural Language Processing.

While in a Bag of Words the vector represents the entire text, with Word Vectors a vector represents just a word, or to be more precise, represents a *meaning*. In fact, Word Vectors can be summed and subtracted to form new meanings, such as the following well know examples:

- "king" - "man" + "woman" = "queen"
- "paris" - "france" + "italy" = "rome"
- "bought" - "buy" + "sell" = "sold"

The Word Vectors toolkit in MAChineLearning supports loading pre-computed word vector dictionaries of the following models:

- [Word2vec (Google)](https://code.google.com/archive/p/word2vec/), both text and binary format.
- [GloVe (Stanford)](http://nlp.stanford.edu/projects/glove/), text format only.
- [fastText (Facebook)](https://github.com/facebookresearch/fastText), text format only.

Note: While a tentative at building a Word Vectors dictionary from a text corpus has been made, using the neural networks of MAChineLearning, it resulted impractically slow. Computing Word Vectors from scratch, in fact, requires code specifically optimized for the task, since each text is a sparse vector (a Bag of Words, actually) and a general purpose neural network wastes lots of time computing values for zeroed elements.


### Tutorial

#### Loading a Word Vectors dictionary

The MLWordVectorDictionary class provides factory methods to load a pre-build dictionary:

```obj-c
// Load a Word2vec dictionary in binary format
MLWordVectorDictionary *dictionary= [MLWordVectorDictionary createFromWord2vecFile:word2vecSamplePath
                                                                     binary:YES];
```


#### Forming meanings with Word Vectors

From the dictionary it is easy to get the Word Vector for a specific word. Each vector provides methods to sum and subtract to/from other vectors, and the dictionary provides methods to search for the nearest word to a vector:

```obj-c
// Philadelphia is related to Pennsylvania as Miami is related to...?
MLWordVector *philadelphia= [map vectorForWord:@"philadelphia"];
MLWordVector *pennsylvania= [map vectorForWord:@"pennsylvania"];
MLWordVector *miami= [map vectorForWord:@"miami"];
MLWordVector *result= [[pennsylvania subtractVector:philadelphia] addVector:miami];

// Search for the word nearest to the resulting vector, hopefully "florida"
NSArray *similarWords= [map mostSimilarWordsToVector:result];
NSLog(@"Result: %@", [similarWords objectAtIndex:0]);
```


#### Using Word Vectors with a neural network

Each Word Vector exposes its full vector as a C buffer (array), ready to be feeded to a neural network:

```obj-c
// Philadelphia is related to Pennsylvania as Miami is related to...?
// ...
MLWordVector *result= [[pennsylvania subtractVector:philadelphia] addVector:miami];

// Clear the input buffer with Accelerate
ML_VCLR(net.inputBuffer, 1, net.inputSize);

// Load the vector in the input buffer with Accelerate
ML_VADD(result.vector, 1, net.inputBuffer, 1, net.inputBuffer, 1, result.size);
// ...

// Run the network
[net feedForward];

// Evaluate the result
// ...
```


### Examples

The framework contains some unit tests that show how to use it, see [WordVectorTests.m](MAChineLearningTests/WordVectorTests.m).


### References

The following two resources may be of help to understand how to use word vectors and how they are produced:

* [Bag of Words Meets Bags of Popcorn - Part 2: Word Vectors](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-2-word-vectors)
* [GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/pubs/glove.pdf) [PDF]


## About me

I am a professional developer but not a data scientist. I wrote this library because, you know, they say you haven't really understood something until you can code it. So, here it is. Use it to experiment and have fun, and if you find it useful I will be happy to hear it at [@self_vs_this](http://www.twitter.com/self_vs_this).

Every effort has been taken to guarantee the framework is error-free, including a side-by-side weights/results comparision with other open source software. If you find bugs or conceptual mistakes please report them. So I can fix them and, most importantly, learn something new.

Enjoy.

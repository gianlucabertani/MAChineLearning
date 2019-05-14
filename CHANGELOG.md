# MAChineLearning Change Log


## 1.0.5

Minor changes:

- Fixed nullability warnings.
- Fixed failing unit test.


## 1.0.4

Minor changes to MLWordVectorDictionary:

- Added backup and restore.
- Added add and remove of words.
- Added support for words that include no-break spaces.


## 1.0.3

Minor changes:

- Improved memory consumption of MLWordVectorDictionary.
- Added generics and nullability annotations.


## 1.0.2

Minor changes:

- Added sentence vectorization in MLWordVectorDictionary
- Improved memory consumption of MLWordVectorDictionary


## 1.0.1

Minor changes:

- Fixed bugs with fastText word vector dictionary loading
- Added unit test for fastText word vectors


## 1.0

Major changes:

- Added support for Word Vectors
- Added support for resilient backpropagation (RPROP)
- Added support for cross entropy cost function
- Added support for rectified linear units (ReLU)
- Added full MNIST example

Minor changes:

- Added `ML` prefix to all classes
- Changed implementation of bias:
  - Networks saved with previous version now are incompatible
- Changed implementation of Bag of Words:
  - Now the dictionary is a specialized object, MLWordDictionary
- Improved unit tests
- Improved, updated and revised README
- Distributed via CocoaPods


## 0.x

First public release.


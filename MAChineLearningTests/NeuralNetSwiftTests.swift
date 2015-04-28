//
//  NeuralNetSwiftTests.swift
//  MAChineLearning
//
//  Created by Gianluca Bertani on 20/04/15.
//  Copyright (c) 2015 Gianluca Bertani. All rights reserved.
//

import Cocoa
import XCTest
import MAChineLearning

let NAND_TEST_TRAIN_CYCLES = 3
let NAND_TEST_LEARNING_RATE: Float = 0.1

let BACKPROPAGATION_TEST_TRAIN_CYCLES = 3000
let BACKPROPAGATION_TEST_LEARNING_RATE: Float = 0.5
let BACKPROPAGATION_TEST_VERIFICATION_CYCLES = 50

let LOAD_SAVE_TEST_TRAIN_CYCLES = 2000
let LOAD_SAVE_TEST_LEARNING_RATE: Float = 0.8

let POW_2_52 = 4503599627370496.0


class NeuralNetSwiftTests: XCTestCase {

    override func setUp() {
        super.setUp()
    }
    
    override func tearDown() {
        super.tearDown()
    }
	
	func testNAND_Swift() {
		let net = NeuralNetwork.createNetworkWithLayerSizes([3, 1], outputFunctionType: ActivationFunctionTypeStep)
		
		let begin = NSDate()
		
		for i in 0...NAND_TEST_TRAIN_CYCLES {
			
			// First set
			net.inputBuffer[0] = 1.0
			net.inputBuffer[1] = 0.0
			net.inputBuffer[2] = 0.0
			net.expectedOutputBuffer[0] = 1.0
			
			net.feedForward()
			net.backPropagateWithLearningRate(NAND_TEST_LEARNING_RATE)
			net.updateWeights()
			
			// Second set
			net.inputBuffer[0] = 1.0
			net.inputBuffer[1] = 0.0
			net.inputBuffer[2] = 1.0
			net.expectedOutputBuffer[0] = 1.0
			
			net.feedForward()
			net.backPropagateWithLearningRate(NAND_TEST_LEARNING_RATE)
			net.updateWeights()
			
			// Third set
			net.inputBuffer[0] = 1.0
			net.inputBuffer[1] = 1.0
			net.inputBuffer[2] = 0.0
			net.expectedOutputBuffer[0] = 1.0
			
			net.feedForward()
			net.backPropagateWithLearningRate(NAND_TEST_LEARNING_RATE)
			net.updateWeights()
			
			// Fourth set
			net.inputBuffer[0] = 1.0
			net.inputBuffer[1] = 1.0
			net.inputBuffer[2] = 1.0
			net.expectedOutputBuffer[0] = 0.0
			
			net.feedForward()
			net.backPropagateWithLearningRate(NAND_TEST_LEARNING_RATE)
			net.updateWeights()
			
			/* Uncomment to dump network status
			
			// Dump network status
			let layer = net.layers[1] as! NeuronLayer
			let neuron = layer.neurons[0] as! Neuron
			NSLog("testNAND: bias: %.2f, weight 1: %.2f, weight 2: %.2f, weight 3: %.2f", neuron.bias, neuron.weights[0], neuron.weights[1], neuron.weights[2])
			
			 */
		}
		
		let elapsed = NSDate().timeIntervalSinceDate(begin)
		NSLog("testNAND_Swift: average training time: %.2f µs per cycle", (elapsed * 1000000.0) / (4.0 * Double(NAND_TEST_TRAIN_CYCLES)))
		
		// First test
		net.inputBuffer[0] = 1.0
		net.inputBuffer[1] = 0.0
		net.inputBuffer[2] = 0.0
		
		net.feedForward()
		
		XCTAssertEqualWithAccuracy(net.outputBuffer[0], 1.0, 0.1)
		
		// Second test
		net.inputBuffer[0] = 1.0
		net.inputBuffer[1] = 0.0
		net.inputBuffer[2] = 1.0
		
		net.feedForward()
		
		XCTAssertEqualWithAccuracy(net.outputBuffer[0], 1.0, 0.1)
		
		// Third test
		net.inputBuffer[0] = 1.0
		net.inputBuffer[1] = 1.0
		net.inputBuffer[2] = 0.0
		
		net.feedForward()
		
		XCTAssertEqualWithAccuracy(net.outputBuffer[0], 1.0, 0.1)
		
		// Fourth test
		net.inputBuffer[0] = 1.0
		net.inputBuffer[1] = 1.0
		net.inputBuffer[2] = 1.0
		
		net.feedForward()
		
		XCTAssertEqualWithAccuracy(net.outputBuffer[0], 0.0, 0.1)
	}
	
	func testBackpropagation_Swift() {
		let net = NeuralNetwork.createNetworkWithLayerSizes([2, 2, 1], outputFunctionType: ActivationFunctionTypeLinear)
		
		let layer1 = net.layers[1] as! NeuronLayer
		let neuron11 = layer1.neurons[0] as! Neuron
		let neuron12 = layer1.neurons[1] as! Neuron
		
		let layer2 = net.layers[2] as! NeuronLayer
		let neuron2 = layer2.neurons[0] as! Neuron
		
		let begin = NSDate()
		
		for i in 1...BACKPROPAGATION_TEST_TRAIN_CYCLES + BACKPROPAGATION_TEST_VERIFICATION_CYCLES {
			let sum = Float(randomDouble(0.10, max: 0.90))
			
			net.inputBuffer[0] = 3.0 * sum / 2.0
			net.inputBuffer[1] = sum / 3.0
			
			net.feedForward()
			
			let computedOutput = net.outputBuffer[0]
			net.expectedOutputBuffer[0] = Float(sum)
			let delta = abs(sum - computedOutput)
			
			if i <= BACKPROPAGATION_TEST_TRAIN_CYCLES {
				net.backPropagateWithLearningRate(BACKPROPAGATION_TEST_LEARNING_RATE)
				net.updateWeights()
				
				/* Uncomment to dump network status
				
				// Dump network status
				NSLog("testBackpropagation_Swift: training cycle %d, expected: %.2f, computed: %.2f, delta: %.2f, status:\n" +
					"\t|B:%.2f W01:%.2f W02:%.2f|\n" +
					"\t                           x |B:%.2f W21:%.2f W22:%.2f|\n" +
					"\t|B:%.2f W01:%.2f W02:%.2f|\n\n",
					i, sum, computedOutput, delta,
					neuron11.bias, neuron11.weights[0], neuron11.weights[1],
					neuron2.bias, neuron2.weights[0], neuron2.weights[1],
					neuron12.bias, neuron12.weights[0], neuron12.weights[1])

				 */
				
			} else {
				
				// Check accuracy in the last cycles
				XCTAssertEqualWithAccuracy(delta, 0.00, 0.05)
			}
		}
		
		let elapsed = NSDate().timeIntervalSinceDate(begin)
		NSLog("testBackpropagation_Swift: average training/verification time: %.2f µs per cycle", (elapsed * 1000000.0) / Double(BACKPROPAGATION_TEST_TRAIN_CYCLES + BACKPROPAGATION_TEST_VERIFICATION_CYCLES))
		
		// Check final weights
		XCTAssertEqualWithAccuracy(neuron11.bias, -0.35, 0.05)
		XCTAssertEqualWithAccuracy(neuron11.weights[0], 1.15, 0.05)
		XCTAssertEqualWithAccuracy(neuron11.weights[1], 0.26, 0.05)
		
		XCTAssertEqualWithAccuracy(neuron12.bias, -0.35, 0.05)
		XCTAssertEqualWithAccuracy(neuron12.weights[0], 1.15, 0.05)
		XCTAssertEqualWithAccuracy(neuron12.weights[1], 0.26, 0.05)
		
		XCTAssertEqualWithAccuracy(neuron2.bias, -1.05, 0.05)
		XCTAssertEqualWithAccuracy(neuron2.weights[0], 1.20, 0.05)
		XCTAssertEqualWithAccuracy(neuron2.weights[1], 1.20, 0.05)
	}
	
	func testLoadSave_Swift() {
		let net = NeuralNetwork.createNetworkWithLayerSizes([3, 2, 1], outputFunctionType: ActivationFunctionTypeLogistic)
		
		let layer1 = net.layers[1] as! NeuronLayer
		let neuron11 = layer1.neurons[0] as! Neuron
		let neuron12 = layer1.neurons[1] as! Neuron
		
		let layer2 = net.layers[2] as! NeuronLayer
		let neuron2 = layer2.neurons[0] as! Neuron
		
		let begin = NSDate()
		
		// Train the network with random numbers
		for i in 1...LOAD_SAVE_TEST_TRAIN_CYCLES {
			net.inputBuffer[0] = Float(randomDouble(0.0, max: 5.0))
			net.inputBuffer[1] = Float(randomDouble(-2.5, max: 2.5))
			net.inputBuffer[2] = Float(randomDouble(-5.0, max: 0.0))
			net.expectedOutputBuffer[0] = Float(randomDouble(0.5, max: 1.0))
			
			net.feedForward()
			net.backPropagateWithLearningRate(LOAD_SAVE_TEST_LEARNING_RATE)
			net.updateWeights()
		}
		
		/* Uncomment to dump network status at end of training
		
		// Dump network status
		NSLog("testLoadSave_Swift: reference network status after %d training cycles:\n" +
			"\t|B:%.2f W01:%.2f W02:%.2f|\n" +
			"\t                           x |B:%.2f W21:%.2f W22:%.2f|\n" +
			"\t|B:%.2f W01:%.2f W02:%.2f|\n\n",
			LOAD_SAVE_TEST_TRAIN_CYCLES,
			neuron11.bias, neuron11.weights[0], neuron11.weights[1],
			neuron2.bias, neuron2.weights[0], neuron2.weights[1],
			neuron12.bias, neuron12.weights[0], neuron12.weights[1])
		
		*/
		
		let elapsed = NSDate().timeIntervalSinceDate(begin)
		NSLog("testLoadSave_Swift: average training time: %.2f µs per cycle", (elapsed * 1000000.0) / Double(LOAD_SAVE_TEST_TRAIN_CYCLES))
		
		// Run a simple compute and save inputs and output
		let input0 = Float(randomDouble(0.0, max: 5.0))
		let input1 = Float(randomDouble(-2.5, max: 2.5))
		let input2 = Float(randomDouble(-5.0, max: 0.0))
		
		net.inputBuffer[0] = input0
		net.inputBuffer[1] = input1
		net.inputBuffer[2] = input2
		
		net.feedForward()
		
		let output = net.outputBuffer[0]
		
		// Save the config and recreate the network
		let config = net.saveConfigurationToDictionary()
		let net2 = NeuralNetwork.createNetworkFromConfigurationDictionary(config)
		
		let layer1_2 = net2.layers[1] as! NeuronLayer
		let neuron11_2 = layer1_2.neurons[0] as! Neuron
		let neuron12_2 = layer1_2.neurons[1] as! Neuron
		
		let layer2_2 = net2.layers[2] as! NeuronLayer
		let neuron2_2 = layer2_2.neurons[0] as! Neuron
		
		/* Uncomment to dump network status after recreation
		
		// Dump network status
		NSLog("testLoadSave_Swift: recreated network status:\n" +
			"\t|B:%.2f W01:%.2f W02:%.2f|\n" +
			"\t                           x |B:%.2f W21:%.2f W22:%.2f|\n" +
			"\t|B:%.2f W01:%.2f W02:%.2f|\n\n",
			neuron11_2.bias, neuron11_2.weights[0], neuron11_2.weights[1],
			neuron2_2.bias, neuron2_2.weights[0], neuron2_2.weights[1],
			neuron12_2.bias, neuron12_2.weights[0], neuron12_2.weights[1])
		
		*/
		
		// Check the weights are the same
		XCTAssertEqualWithAccuracy(neuron11_2.bias, neuron11.bias, 0.0000000001)
		XCTAssertEqualWithAccuracy(neuron11_2.weights[0], neuron11.weights[0], 0.0000000001)
		XCTAssertEqualWithAccuracy(neuron11_2.weights[1], neuron11.weights[1], 0.0000000001)
		
		XCTAssertEqualWithAccuracy(neuron12_2.bias, neuron12.bias, 0.0000000001)
		XCTAssertEqualWithAccuracy(neuron12_2.weights[0], neuron12.weights[0], 0.0000000001)
		XCTAssertEqualWithAccuracy(neuron12_2.weights[1], neuron12.weights[1], 0.0000000001)
		
		XCTAssertEqualWithAccuracy(neuron2_2.bias, neuron2.bias, 0.0000000001)
		XCTAssertEqualWithAccuracy(neuron2_2.weights[0], neuron2.weights[0], 0.0000000001)
		XCTAssertEqualWithAccuracy(neuron2_2.weights[1], neuron2.weights[1], 0.0000000001)
		
		// If everything has been saved correctly, submitting the same input should give the same output
		net2.inputBuffer[0] = input0
		net2.inputBuffer[1] = input1
		net2.inputBuffer[2] = input2
		
		net2.feedForward()
		
		XCTAssertEqualWithAccuracy(net2.outputBuffer[0], output, 0.0000000001)
	}

	/*
     * Trick of "withUnsafeMutablePointer" taken from:
     * https://medium.com/@stefanvdoord/secrandomcopybytes-or-playing-unsafe-with-swift-e627397e1b11
     * Thanks @stefanvdoord, you saved my day! I would have never guessed this.
     */
	
	func randomInt(max: UInt) -> UInt {
		var random: UInt = 0
		
		withUnsafeMutablePointer(&random, { (randomNumberPointer) -> Void in
			var castedPointer = unsafeBitCast(randomNumberPointer, UnsafeMutablePointer<UInt8>.self)
			SecRandomCopyBytes(kSecRandomDefault, sizeof(UInt), castedPointer)
		})
	
		return (random % max)
	}
	
	func randomDouble(min: Double, max: Double) -> Double {
		var random : Int64 = 0
		
		withUnsafeMutablePointer(&random, { (randomNumberPointer) -> Void in
			var castedPointer = unsafeBitCast(randomNumberPointer, UnsafeMutablePointer<UInt8>.self)
			SecRandomCopyBytes(kSecRandomDefault, sizeof(Int64), castedPointer)
		})
	
		let rnd = Double(abs(random) >> 11) / POW_2_52
		return min + (rnd * (max - min))
	}
}

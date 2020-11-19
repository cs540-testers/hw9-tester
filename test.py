__author__ = 'cs540-testers'
__credits__ = ['Harrison Clark', 'Stephen Jasina', 'Saurabh Kulkarni',
		'Alex Moon']
version = 'v0.1' 
# TODO: set this to 1.0 once you've independently confirmed output vals

import unittest
import io
import intro_keras
import sys
import numpy as np
from tensorflow import keras

class TestIntroKeras(unittest.TestCase):


    def test_get_dataset(self):
        (train_images, train_labels) = intro_keras.get_dataset()
        (test_images, test_labels) = intro_keras.get_dataset(training=False)

        self.assertEqual(type(train_images), np.ndarray)
        self.assertEqual(type(train_labels), np.ndarray)
        self.assertEqual(type(test_images), np.ndarray)
        self.assertEqual(type(test_labels), np.ndarray)
        self.assertEqual(type(train_labels[0]), np.uint8)
        self.assertEqual(type(train_images[0]), np.ndarray)
        self.assertEqual(train_images.shape, (60000, 28, 28))
        self.assertEqual(train_labels.shape, (60000,))
        self.assertEqual(test_images.shape, (10000, 28, 28))
        self.assertEqual(test_labels.shape, (10000,))

        self.assertEqual(train_labels[0], 5)
        self.assertEqual(test_labels[0], 7)


    def test_print_stats(self):
        (train_images, train_labels) = intro_keras.get_dataset()
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        intro_keras.print_stats(train_images, train_labels)
        output = capturedOutput.getvalue()
        sys.stdout = sys.__stdout__
        expectedoutput = "60000\n"\
        + "28x28\n"\
        + "0. Zero - 5923\n"\
        + "1. One - 6742\n"\
        + "2. Two - 5958\n"\
        + "3. Three - 6131\n"\
        + "4. Four - 5842\n"\
        + "5. Five - 5421\n"\
        + "6. Six - 5918\n"\
        + "7. Seven - 6265\n"\
        + "8. Eight - 5851\n"\
        + "9. Nine - 5949\n"
        self.assertEqual(output, expectedoutput)

    def test_build_model(self):
        # TODO: use model.get_layer to check layers?
        # Kinda checked in train_model
        model = intro_keras.build_model()
        self.assertEqual(model.__class__.__name__, "Sequential")
        self.assertEqual(model.loss.__class__.__name__, "SparseCategoricalCrossentropy")
        self.assertEqual(model.optimizer.__class__.__name__, "SGD")
        self.assertEqual(model.metrics_names, []) #should be empty before fit() is called


    
    def test_train_model(self):
        (train_images, train_labels), (
        test_images, test_labels) = keras.datasets.mnist.load_data()
        model = intro_keras.build_model()
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        intro_keras.train_model(model, train_images, train_labels, 10)
        output = capturedOutput.getvalue()
        sys.stdout = sys.__stdout__

        expected_summary = [
            "_________________________________________________________________",
            "Layer (type)                 Output Shape              Param #   ",
            "=================================================================",
            "flatten_3 (Flatten)          (32, 784)                 0         ",
            "_________________________________________________________________",
            "dense_9 (Dense)              (32, 128)                 100480    ",
            "_________________________________________________________________",
            "dense_10 (Dense)             (32, 64)                  8256      ",
            "_________________________________________________________________",
            "dense_11 (Dense)             (32, 10)                  650       ",
            "=================================================================",
            "Total params: 109,386",
            "Trainable params: 109,386",
            "Non-trainable params: 0",
            "_________________________________________________________________",
            ""]
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        model.summary()#prints to stdio
        output = capturedOutput.getvalue()
        sys.stdout = sys.__stdout__

        # remove model name because it's numbered
        summary_lines = output.split('\n')[1:]

        for i, (ref, real) in enumerate(zip(expected_summary, summary_lines)):
            if i in (3,5,7,9):
                # layer names are numbered according to the order
                # you run the tests in
                ref = "".join(ref.split(" ")[1:])
                real = "".join(real.split(" ")[1:])
            self.assertEqual(ref.rstrip(), real.rstrip())


    
    def test_evaluate_model(self):
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        (train_images, train_labels), (
        test_images, test_labels) = keras.datasets.mnist.load_data()
        model = intro_keras.build_model()
        intro_keras.train_model(model, train_images, train_labels, 10)
        intro_keras.evaluate_model(model, test_images, test_labels)
        output = capturedOutput.getvalue()
        sys.stdout = sys.__stdout__

        # splice only the evaluate part
        summary_lines = output.split('\n')[21:]
        loss_line = summary_lines[0]
        loss = float(loss_line.split(":")[1].strip())
        self.assertAlmostEqual(loss, 0.2, delta=0.05)
        accuracy_line = summary_lines[1]
        accuracy = float(accuracy_line.split(":")[1].strip().strip('%'))
        self.assertAlmostEqual(accuracy, 94.5, delta=1.0)



    def test_predict_label(self):
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        (train_images, train_labels), (
        test_images, test_labels) = keras.datasets.mnist.load_data()
        model = intro_keras.build_model()
        intro_keras.train_model(model, train_images, train_labels, 10)
        intro_keras.evaluate_model(model, test_images, test_labels)
        model.add(keras.layers.Softmax())
        intro_keras.predict_label(model, test_images, 1)

        output = capturedOutput.getvalue().split('\n')
        sys.stdout = sys.__stdout__

        # splice out the non-prediction stuff
        output = output[23:]

        best_pred = output[0].split(':')[0]
        self.assertEqual(best_pred, "Two")

    
if __name__ == "__main__":
    unittest.main()

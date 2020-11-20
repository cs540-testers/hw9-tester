__author__ = 'cs540-testers'
__credits__ = ['Harrison Clark', 'Stephen Jasina', 'Saurabh Kulkarni',
		'Alex Moon']
__version__ = '1.1'

# Suppress GPU warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import contextlib
import io
import unittest
import numpy as np
import tensorflow as tf
from tensorflow import keras
from intro_keras import get_dataset, print_stats, build_model, train_model, \
        evaluate_model, predict_label

class TestIntroKeras(unittest.TestCase):
    def test_get_dataset(self):
        train_images, train_labels = get_dataset()
        test_images, test_labels = get_dataset(False)

        self.assertIsInstance(train_images, np.ndarray)
        self.assertIsInstance(train_labels, np.ndarray)
        self.assertIsInstance(test_images, np.ndarray)
        self.assertIsInstance(test_labels, np.ndarray)
        self.assertIsInstance(train_labels[0], np.uint8)
        self.assertIsInstance(train_images[0], np.ndarray)
        self.assertEqual(train_images.shape, (60000, 28, 28))
        self.assertEqual(train_labels.shape, (60000,))
        self.assertEqual(test_images.shape, (10000, 28, 28))
        self.assertEqual(test_labels.shape, (10000,))

        self.assertEqual(train_labels[0], 5)
        self.assertEqual(test_labels[0], 7)


    def test_print_stats(self):
        train_images, train_labels = get_dataset()
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            print_stats(train_images, train_labels)
        expectedoutput = '60000\n'\
                + '28x28\n'\
                + '0. Zero - 5923\n'\
                + '1. One - 6742\n'\
                + '2. Two - 5958\n'\
                + '3. Three - 6131\n'\
                + '4. Four - 5842\n'\
                + '5. Five - 5421\n'\
                + '6. Six - 5918\n'\
                + '7. Seven - 6265\n'\
                + '8. Eight - 5851\n'\
                + '9. Nine - 5949\n'
        self.assertEqual(captured_output.getvalue(), expectedoutput)


    def test_build_model(self):
        model = build_model()

        # Check model type
        self.assertIsInstance(model,
                tf.python.keras.engine.sequential.Sequential)
        self.assertIsInstance(model.loss,
                tf.python.keras.losses.SparseCategoricalCrossentropy)
        self.assertIsInstance(model.optimizer,
                tf.python.keras.optimizer_v2.gradient_descent.SGD)

        # Check layers
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            model.summary()
        expected_summary = [
            'Model: "sequential"',
            '_________________________________________________________________',
            'Layer (type)                 Output Shape              Param #',
            '=================================================================',
            'flatten (Flatten)            (None, 784)               0',
            '_________________________________________________________________',
            'dense (Dense)                (None, 128)               100480',
            '_________________________________________________________________',
            'dense_1 (Dense)              (None, 64)                8256',
            '_________________________________________________________________',
            'dense_2 (Dense)              (None, 10)                650',
            '=================================================================',
            'Total params: 109,386',
            'Trainable params: 109,386',
            'Non-trainable params: 0',
            '_________________________________________________________________'
        ]

        summary_lines = captured_output.getvalue().split('\n')
        for i in (4, 6, 8, 10):
            # Layer names are numbered according to the order you run the tests,
            # so we have to omit them from the comparison
            real = ''.join(summary_lines[i].split(' ')[1:])
            ref = ''.join(expected_summary[i].split(' ')[1:])
            self.assertEqual(real, ref)


    def test_train_model(self):
        train_images, train_labels = get_dataset()
        model = build_model()

        # Check that the number of samples is printed
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            train_model(model, train_images, train_labels, 10)
        output_lines = captured_output.getvalue().split('\n')
        self.assertEqual(len(output_lines), 21,
                'Incorrect number of lines printed')
        self.assertEqual(output_lines[-1], '', 'Output should end in newline')
        for i in range(10):
            self.assertEqual(output_lines[2 * i], 'Epoch {:d}/10'.format(i + 1))

        # If this test fails, you can probably(?) comment it out
        self.assertEqual(model.metrics_names, ['loss', 'accuracy'])


    def test_evaluate_model(self):
        train_images, train_labels = get_dataset()
        test_images, test_labels = get_dataset(False)
        model = build_model()
        with contextlib.redirect_stdout(io.StringIO()):
            train_model(model, train_images, train_labels, 10)

        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            evaluate_model(model, test_images, test_labels)

        # Splice only the evaluate part
        summary_lines = captured_output.getvalue().split('\n')
        # Check that nothing extra is printed (note that 3 is used here because
        # the last newline character causes an empty string to be the last
        # element of summary_lines)
        self.assertEqual(len(summary_lines), 3,
                'Printed incorrect number of lines')
        self.assertEqual(summary_lines[-1], '')

        loss_line = summary_lines[0]
        self.assertEqual(loss_line[:8], 'Loss: 0.')
        self.assertEqual(len(loss_line), 12, 'Loss needs to have 4 decimals')
        self.assertAlmostEqual(float(loss_line[6:]), 0.2, delta=0.05)

        accuracy_line = summary_lines[1]
        self.assertEqual(accuracy_line[:10], 'Accuracy: ')
        self.assertEqual(accuracy_line[-1], '%')
        self.assertEqual(len(accuracy_line), 16,
                'Accuracy needs to have 2 decimals')
        self.assertAlmostEqual(float(accuracy_line[10:15]), 94.5, delta=1.0)

        # Repeat the test, but only print the accuracy
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            evaluate_model(model, test_images, test_labels, False)

        summary_lines = captured_output.getvalue().split('\n')
        self.assertEqual(len(summary_lines), 2,
                'Printed incorrect number of lines')
        self.assertEqual(summary_lines[-1], '')

        accuracy_line = summary_lines[0]
        self.assertEqual(accuracy_line[:10], 'Accuracy: ')
        self.assertEqual(accuracy_line[-1], '%')
        self.assertEqual(len(accuracy_line), 16,
                'Accuracy needs to have 2 decimals')
        self.assertAlmostEqual(float(accuracy_line[10:15]), 94.5, delta=1.0)


    def test_predict_label(self):
        train_images, train_labels = get_dataset()
        test_images, test_labels = get_dataset(False)
        model = build_model()
        with contextlib.redirect_stdout(io.StringIO()):
            train_model(model, train_images, train_labels, 10)
            evaluate_model(model, test_images, test_labels)

        model.add(keras.layers.Softmax())
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            predict_label(model, test_images, 1)

        predictions = captured_output.getvalue().split('\n')
        self.assertEqual(len(predictions), 4, 'Must print 3 predictions')
        self.assertEqual(predictions[-1], '')

        # Pick the piece of the output corresponding to the best prediction
        best_prediction = predictions[0]
        self.assertEqual(best_prediction[:5], 'Two: ')
        self.assertEqual(best_prediction[-1], '%')
        self.assertTrue(len(best_prediction) == 11
                or best_prediction == 'Two: 100.00%',
                'Prediction needs to have 2 decimals')


if __name__ == '__main__':
    print('Homework 9 Tester Version', __version__)
    unittest.main()

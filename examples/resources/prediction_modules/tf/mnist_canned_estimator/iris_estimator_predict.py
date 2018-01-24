import tensorflow as tf
import numpy as np


def load_model(model_dir):
    return TensorflowMnistEstimator(model_dir)


class TensorflowMnistEstimator:

    def __init__(self, model_dir):
        self.feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]
        self.classifier = tf.estimator.DNNClassifier(feature_columns=self.feature_columns,
                                                     hidden_units=[10, 20, 10],
                                                     n_classes=3,
                                                     model_dir=model_dir)

    def predict(self, given_input):
        new_samples = np.array(
            given_input, dtype=np.float32)
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": new_samples},
            num_epochs=1,
            shuffle=False)

        predictions = list(self.classifier.predict(input_fn=predict_input_fn))
        predicted_classes = [int(p["classes"].tolist()[0]) for p in predictions]
        return predicted_classes




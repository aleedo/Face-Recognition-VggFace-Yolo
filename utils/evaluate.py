import tensorflow as tf
from utils.const import MODEL_PATH

def load_model(path=MODEL_PATH):
    model = tf.keras.models.load_model(path)
    return model


def eval_model(test_dataset, path=MODEL_PATH):
    model = load_model()
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Testing:\nLoss: {test_loss}\nAccuracy: {test_acc}")
    return test_loss, test_acc
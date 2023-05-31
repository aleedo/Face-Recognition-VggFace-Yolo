from utils.const import BATCH_SIZE, IMAGE_SIZE
import tensorflow_datasets as tfds
from keras_vggface.utils import preprocess_input
import tensorflow as tf


def get_data():
    lfw_dataset, info = tfds.load('lfw', split='train', with_info=True, as_supervised=True)
    return lfw_dataset, info

def preprocess(label, image):
    # image = preprocess_input(image)
    image = tf.image.resize(image, IMAGE_SIZE) / 255.0

    num_label = tf.strings.to_hash_bucket_fast(label, NUM_CLASSES)
    return image, num_label

def get_train_val_test_splits(ds):
    # Setup for train dataset
    val_dataset = ds.take(100)
    test_dataset = ds.skip(100).take(100)
    train_dataset = ds.skip(100)

    train_dataset = train_dataset.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    val_dataset = val_dataset.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    test_dataset = test_dataset.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Shuffle and batch the datasets
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.cache()
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    test_dataset = test_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.cache()
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset


def get_labels(ds):
    person_names = set()
    for example in tfds.as_numpy(ds):
        person_name = example[0].decode("utf-8")
        person_names.add(person_name)

    # Get the number of unique person names
    num_classes = len(person_names)
    print(f"Number of classes in the dataset: {num_classes}")

    person_names = sorted(person_names)
    label_map = dict(zip(person_names, range(num_classes)))
    inverse_label_map = dict(zip(range(num_classes), person_names))

    return person_names, num_classes, label_map, inverse_label_map


labels, NUM_CLASSES, input_output_map, output_input_map = get_labels(get_data()[0])
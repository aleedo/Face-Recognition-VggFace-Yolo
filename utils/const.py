from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# Define the image size for the dataset
IMAGE_SIZE = (224, 224)

# Define the batch size for training and validation
BATCH_SIZE = 32

# Define Number of Epochs for training and validation
EPOCHS = 20

MODEL_PATH = 'src/tf_model/tuned_vgg_face'

EARLY_STOP = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=False
)

CHECK_POINT = ModelCheckpoint('weights.hdf5', monitor='val_loss', save_best_only=True)

CALL_BACKS = [
    EARLY_STOP, 
    CHECK_POINT
]
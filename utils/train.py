
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten
from utils.const import CALL_BACKS, EPOCHS, CALL_BACKS, MODEL_PATH
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from utils.load_preprocess import NUM_CLASSES


def save_model(model):
    model.save(MODEL_PATH)

def fit_model(train_dataset, val_dataset):
    base_model = VGGFace(include_top=False, weights='vggface', input_tensor=None, input_shape=[224, 224, 3]) 
    # Add a custom output layer to the VGGFace model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze all layers in the VGGFace model except the custom output layer
    for layer in base_model.layers:
        layer.trainable = False
    for layer in model.layers[-2:]:
        layer.trainable = True

    # Compile the model with the Adam optimizer, sparse categorical crossentropy loss function, and accuracy metric
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model on the training dataset and validate on the validation dataset
    history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=CALL_BACKS)

    # Save the model
    save_model(model)

    return history
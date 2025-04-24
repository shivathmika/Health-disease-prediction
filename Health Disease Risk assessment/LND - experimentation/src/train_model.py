from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from model_architecture import buildCNNModel
from config import SAVE_MODEL_PATH
import keras

def train_model(x, y, input_shape=(64, 64, 1), num_classes=2, epochs=50, batch_size=100):
    # Split into train and validation sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
 
    y_train = keras.utils.to_categorical(y_train, num_classes=2)
    y_test = keras.utils.to_categorical(y_test, num_classes=2)
    # Build and compile the model
    model = buildCNNModel()
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='Adam',metrics=['accuracy'])
    # Train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)

    return model


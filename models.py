from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class Models:
    def createCNN(*args):
        inputs = layers.Input(shape=args[1])
        model = layers.Conv2D(64, 3, kernel_initializer='he_normal', strides = 1, padding = "same", activation="relu")(inputs)
        model = layers.MaxPooling2D((1,2))(model)
        model = layers.Conv2D(128, 3, strides = 1, padding = "same", activation="relu")(model)
        model = layers.MaxPooling2D((1,2))(model)
        model = layers.Conv2D(128, 3, strides = 1, padding = "same", activation="relu")(model)
        model = layers.MaxPooling2D((1,2))(model)
        model = layers.Conv2D(128, 3, strides = 1, padding = "same", activation="relu")(model)
        model = layers.MaxPooling2D((1,2))(model)
        model = layers.Flatten()(model)
        class_output = layers.Dense(args[2], activation="softmax", name="class_output")(model)

        model = Model(inputs=[inputs], outputs=[class_output], name="CNN_Tree")
        optimizer = Adam(lr=0.00001)
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                        optimizer=optimizer,metrics=["accuracy"])
            
        return model

    def createDense(*args):
        inputs = layers.Input(shape=args[1])
        model = layers.Flatten()(inputs)
        model = layers.Dense(256, activation="relu")(model)
        model = layers.Dense(128, activation="relu")(model)
        model = layers.Dense(64, activation="relu")(model)
        model = layers.Dense(32, activation="relu")(model)
        model = layers.Dense(16, activation="relu")(model)
        class_output = layers.Dense(args[2], activation="softmax", name="class_output")(model)

        model = Model(inputs=[inputs], outputs=[class_output], name="DenseTreeSeq")
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), 
                      optimizer="adam",metrics=["accuracy"])
        
        return model

    def createRF():
        pass

    def createMNIST():
        model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model
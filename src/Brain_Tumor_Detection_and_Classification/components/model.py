import tensorflow as tf
import sys
from tensorflow.keras import layers as L
from src.Brain_Tumor_Detection_and_Classification.exception import CustomException
  
class Model:
    def cnn_model(self):
        try:        
            input_shape=(150,150,3)
            inputs = L.Input(input_shape)
            x = L.Conv2D(filters=32,kernel_size=3, activation = "relu",padding="same")(inputs)
            x = L.Conv2D(filters=64,kernel_size=3, activation = "relu",padding="same")(x)
            x = L.MaxPool2D(pool_size=(2,2))(x)
            x = L.Dropout(0.3)(x)

            x = L.Conv2D(filters=64,kernel_size=3, activation = "relu",padding="same")(x)
            x = L.Conv2D(filters=64,kernel_size=3, activation = "relu",padding="same")(x)
            x = L.Dropout(0.3)(x)
            x = L.MaxPool2D(pool_size=(2,2))(x)
            x = L.Dropout(0.3)(x)

            x = L.Conv2D(filters=128,kernel_size=3, activation = "relu",padding="same")(x)
            x = L.Conv2D(filters=128,kernel_size=3, activation = "relu",padding="same")(x)
            x = L.Conv2D(filters=128,kernel_size=3, activation = "relu",padding="same")(x)
            x = L.MaxPool2D(pool_size=(2,2))(x)
            x = L.Dropout(0.3)(x)

            x = L.Conv2D(filters=128,kernel_size=3, activation = "relu",padding="same")(x)
            x = L.Conv2D(filters=256,kernel_size=3, activation = "relu",padding="same")(x)
            x = L.MaxPool2D(pool_size=(2,2))(x)
            x= L.Dropout(0.3)(x)

            outputs = L.Dense(4, activation="softmax")(x)

            model = tf.keras.models.Model(inputs, outputs)
            
            return (
                model
            )
            
        except Exception as e:
            raise CustomException(e,sys)








        x = L.Flatten()(x)
        x = L.Dense(512, activation = "relu")(x)

        x = L.Dense(512, activation = "relu")(x)
        x= L.Dropout(0.3)(x)

        outputs = L.Dense(4, activation="softmax")(x)

        model = tf.keras.models.Model(inputs, outputs)
        
        model.summary()
from src.Brain_Tumor_Detection_and_Classification.logger import logging
from src.Brain_Tumor_Detection_and_Classification.exception import CustomException
from src.Brain_Tumor_Detection_and_Classification.components.data_ingestion_and_transformation import DataIngestion
from src.Brain_Tumor_Detection_and_Classification.components.model import Model
from src.Brain_Tumor_Detection_and_Classification.components.model_trainer import ModelTrainer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.preprocessing import image
import sys
import cv2

if __name__ == "__main__":
    logging.info("The execution has started")
    
    try:
        dataIngestion = DataIngestion()
        x_train,x_test,y_train,y_test=dataIngestion.initiate_data_ingestion_and_transformation()
        
        modelObject = Model()
        model = modelObject.cnn_model()
        
        modelTrainerObject = ModelTrainer()
        history = modelTrainerObject.initiate_model_training(model,x_train,y_train)
        
        
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        epochs = range(len(acc))
        fig = plt.figure(figsize=(14,7))
        plt.plot(epochs,acc,'r',label="Training Accuracy")
        plt.plot(epochs,val_acc,'b',label="Validation Accuracy")
        plt.legend(loc='upper left')
        plt.show()
        
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(loss))
        fig = plt.figure(figsize=(14,7))
        plt.plot(epochs,loss,'r',label="Training loss")
        plt.plot(epochs,val_loss,'b',label="Validation loss")
        plt.legend(loc='upper left')
        plt.show()
        
        img = cv2.imread('/content/drive/MyDrive/CNN Implementation/dataset/Testing/meningioma_tumor/image(62).jpg')
        img = cv2.resize(img,(150,150))
        img_array = np.array(img)        
        img_array = img_array.reshape(1,150,150,3)
        
        img = image.load_img('/content/drive/MyDrive/CNN Implementation/dataset/Testing/meningioma_tumor/image(62).jpg')
        plt.imshow(img,interpolation='nearest')
        plt.show()
        labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
        a=model.predict(img_array)
        indices = a.argmax()
        print(indices)
        print("tumor type :", labels[indices])

        
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)


from cProfile import label
import logging
import os
import numpy as np
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split 
from src.Brain_Tumor_Detection_and_Classification.exception import CustomException
import sys
import tensorflow as tf
    
class DataIngestion:    
    def initiate_data_ingestion_and_transformation(self): 
        try: 
            
            logging.info("Data ingestion and Transformation started")      
            drive_path = 'dataset'
            for dirname, _, filenames in os.walk('drive_path'):
                for filename in filenames:
                    print(os.path.join(dirname, filename))
                
            X_train = []
            Y_train = []
            image_size = 150
            labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
            for i in labels:
                folderPath = os.path.join('dataset/Training',i)
                for j in os.listdir(folderPath):
                    img = cv2.imread(os.path.join(folderPath,j))
                    img = cv2.resize(img,(image_size,image_size))
                    X_train.append(img)
                    Y_train.append(i)
                

            for i in labels:
                folderPath = os.path.join('dataset/Testing',i)
                for j in os.listdir(folderPath):
                    img = cv2.imread(os.path.join(folderPath,j))
                    img = cv2.resize(img,(image_size,image_size))
                    X_train.append(img)
                    Y_train.append(i)

            X_train = np.array(X_train)
            Y_train = np.array(Y_train)

            X_train,Y_train = shuffle(X_train,Y_train,random_state=101)

            X_train,X_test,y_train,y_test = train_test_split(X_train,Y_train,test_size=0.1,random_state=101)

            labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
            y_train_new = []
            for i in y_train:
                y_train_new.append(labels.index(i))
            y_train=y_train_new
            y_train = tf.keras.utils.to_categorical(y_train)

            y_test_new = []
            for i in y_test:
                y_test_new.append(labels.index(i))
            y_test=y_test_new
            y_test = tf.keras.utils.to_categorical(y_test)
            
            return (
                X_train,X_test,y_train,y_test
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        
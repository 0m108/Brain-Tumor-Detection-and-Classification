import sys
from src.Brain_Tumor_Detection_and_Classification.exception import CustomException
from src.Brain_Tumor_Detection_and_Classification.components.model import Model
from src.Brain_Tumor_Detection_and_Classification.exception import CustomException



class ModelTrainer:
    def initiate_model_training(self,model,X_train,y_train):
        try:
            model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
            history = model.fit(X_train,y_train,epochs=20,validation_split=0.1)
            
            return(
                history
            )
        except Exception as e:
            raise CustomException(e,sys)
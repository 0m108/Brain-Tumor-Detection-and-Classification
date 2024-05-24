# Brain-Tumor-Detection-and-Classification
A brain tumor, also recognized as an
intracranial tumor, develops as an anomalous mass of
tissue wherein cells proliferate and multiply without
restraint. Detection and classification of tumor in the
brain is of paramount importance for effective
treatment of the patient. CNNs and Vision
Transformers are the popular techniques that have
been used in the past for classification of images. Both
the techniques have performed extraordinarily in the
task of image classification. So here we have
introduced a new approach where we fused VGG16
with the Vision Transformer for brain tumor
detection and classification.
## Dataset
The dataset that we used here contains the MR Images of
Human Brain and it is taken from Kaggle website.
The dataset contains 7023 MRIs, divided into 4 classes:
1. Glioma
2. Pituitary
3. No Tumor
4. Meningioma
For training our model 90% data is used and for testing
10% data is used.
## Flowchart
Life cycle of our approach
![Flow Chart](https://github.com/0m108/Brain-Tumor-Detection-and-Classification/assets/113771586/ac2d3896-9c1e-495c-a6fb-11b85811a462)
## Model Performance
Training Accuracy, Validation Accuracy VS Epochs
![Training acc val acc vs epoch](https://github.com/0m108/Brain-Tumor-Detection-and-Classification/assets/113771586/cb5d926d-3364-48ea-a3d3-abdf3150ddf2)

Training Loss, Validation Loss VS Epochs
![Training loss val loss vs epochs](https://github.com/0m108/Brain-Tumor-Detection-and-Classification/assets/113771586/5c00652f-990c-4430-9b98-0566bfaa3852)
## Installation
To setup the project locally, follow these steps:
1. Clone the repository:
```
https://github.com/0m108/CreditCardFraudDetection.git
cd CreditCardFraudDetection
```
2. Create and activate a virtual environment:
```
python -m venv venv
source venv/bin/activate
```
3. Install the required packages:
```
pip install -r requirements.txt
```
## Usage
To run the model follow this step:
```
python app.py
```



#file created 23:25 04-09-2020
#execution script
#loads model and makes predictions

from train import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#Welcome Note
print("NewSense - Fake News Detection")
print("Enter your choice :")
print("\t1.Train the model")
print("\t2.Make a prediction")
choice=input()
if choice=='1':
    trainer()
elif choice=='2':
    #predictor=load_model()
    pass
    
    

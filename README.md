# DL-Malosite

Deep learning model for prediction of Malonylation sites developed in KC lab by Niraj Thapa.

# Requirement
  Backend = Tensorflow <br/>
  Keras <br/>
  Numpy <br/>
  Biopython <br/>
  Sklearn <br/>
  Imblearn <br/>
 # Dataset
 Dataset is in FASTA format which includes protein window size of 29. Test dataset is provided. There are two dataset for positive and negative examples. Dataset includes all organisms, human, mouse and S-organism. They are provided in separate folder with separate models.
 # Model
 The best model has been included. Download all model parts and extracting the 1st part will automatically extract others. The final file is named as *.h5 (* Model Name)
 # Prediction for given test dataset
 With all the prerequisite installed, run -> model_load.py
 # Prediction for your dataset
 The format should be same as the test dataset which is basically FASTA format. This model works for window size 29 only. 
 # Contact 
 Feel free to contact us if you need any help : nirajthapa@gmail.com, dukka.kc@wichita.edu

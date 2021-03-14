import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Lambda
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from Bio import SeqIO
from numpy import array
from numpy import argmax
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle
from keras.layers.embeddings import Embedding
from keras import backend as K
from keras.backend import expand_dims
import matplotlib.pyplot as plt
from keras.regularizers import l1, l2
from sklearn.metrics import roc_curve, auc, classification_report
from keras.models import load_model


r_test_x = []
r_test_y = []
posit_1 = 1;
negat_0 = 0;
win_size = 29 # actual window size
win_size_kernel = int(win_size/2 + 1)


# define universe of possible input values
alphabet = 'ARNDCQEGHILKMFPSTWYV'
# define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))



#-------------------------TEST DATASET----------------------------------------
#for positive sequence
def innertest1():
    #Input
    data = seq_record.seq
    #rint(data) 
    # integer encode input data
    for char in data:
        if char not in alphabet:
            return
    integer_encoded = [char_to_int[char] for char in data]
    r_test_x.append(integer_encoded)
    r_test_y.append(posit_1)
for seq_record in SeqIO.parse("M-test-pos-107.fasta", "fasta"): # Input positive fasta file
    innertest1()
#for negative sequence
def innertest2():
    #Input
    data = seq_record.seq
    #print(data) 
    # integer encode input data
    for char in data:
        if char not in alphabet:
            return
    integer_encoded = [char_to_int[char] for char in data]
    r_test_x.append(integer_encoded)
    r_test_y.append(negat_0)
for seq_record in SeqIO.parse("M-test-neg-6698.fasta", "fasta"): # Input negative fasta file
    innertest2()
# Changing to array (matrix)    
r_test_x = array(r_test_x)
r_test_y = array(r_test_y)


# Balancing test dataset
# Testing Data Balancing by undersampling####################################
rus = RandomUnderSampler(random_state=7)
x_res3, y_res3 = rus.fit_resample(r_test_x, r_test_y)
#Shuffling
r_test_x, r_test_y = shuffle(x_res3, y_res3, random_state=7)
r_test_x = np.array(r_test_x)
r_test_y = np.array(r_test_y)
############################################################################


##LOAD MODEL####
model = load_model('m1.h5')


from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
Y_pred = model.predict(r_test_x)
t_pred2 = Y_pred[:,1]
Y_pred = (Y_pred > 0.5)
y_pred1 = [np.argmax(y, axis=None, out=None) for y in Y_pred]
y_pred1 = np.array(y_pred1)

print("Matthews Correlation : ",matthews_corrcoef(r_test_y, y_pred1))
print("Confusion Matrix : \n",confusion_matrix(r_test_y, y_pred1))


# For sensitivity and specificity
sp_1, sn_1 = confusion_matrix(r_test_y, y_pred1)
sp_2 = sp_1[0]/(sp_1[0]+sp_1[1])
sn_2 = sn_1[1]/(sn_1[0]+sn_1[1])

# ROC

fpr, tpr, _ = roc_curve(r_test_y, t_pred2)
roc_auc = auc(fpr, tpr)
print("AUC : ", roc_auc)
print(classification_report(r_test_y, y_pred1))

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()

print(model.summary())
print("Specificity = ",sp_2, " Sensitivity = ",sn_2)


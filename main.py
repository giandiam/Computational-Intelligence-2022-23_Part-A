import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import OneHotEncoder

# print floating point numbers using fixed point notation
np.set_printoptions(suppress=True)

harDataset_file_path = r"C:\Users\johnd\ΜΑΘΗΜΑΤΑ\ΥΠΟΛΟΓΙΣΤΙΚΗ ΝΟΗΜΟΣΥΝΗ\Project_ΥΝ_2022-23_Μέρος-Α\dataset-HAR-PUC-Rio.csv"

# read the data, store data in DataFrame titled harDataset_data 
# skip row 122077 having junks in z4 value 
harDataset_data = pd.read_csv(harDataset_file_path, on_bad_lines='skip', delimiter=';', decimal=",",
                            dtype={'user': 'str', 'gender':'str', 'age':'int', 'HowTallInMeters':'float', 
                                   'weight':'int', 'BodyMassIndex':'float', 
                                   'x1':'int', 'y1':'int', 'z1':'int', 'x2':'int', 'y2':'int', 'z2':'int', 
                                   'x3':'int', 'y3':'int', 'z3':'int', 'x4':'int', 'y4':'int', 'z4':'int', 
                                   'class':'str'}, skiprows=[122077])
harDataset_array = np.array(harDataset_data)
harDataset_df = pd.DataFrame(
harDataset_array,
index=list(range(len(harDataset_data))),
columns=['User', 'Gender', 'Age', 'HowTallInMeters', 'Weight', 'BodyMassIndex', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4', 'Class']
)

# debora: 0, katia: 1, wallace: 2, jose_carlos: 3
harDataset_df['User'] = harDataset_df['User'].map({'debora': 0, 'katia': 1, 'wallace': 2, 'jose_carlos': 3})
# Male: 0, Female: 1 mapping
harDataset_df['Gender'] = harDataset_df['Gender'].map({'Man': 0, 'Woman': 1})
# sitting-down: 1, standing-up: 2, standing: 3, walking: 4, sitting: 5
harDataset_df['Class'] = harDataset_df['Class'].map({'sittingdown': 1, 'standingup': 2, 'standing': 3, 'walking': 4, 'sitting': 5})

##################################### NORMALIZATION ###############################################
# 0 to 1 scaling on fields (except 'User', 'Gender')
#harDataset_df[['User','Gender', 'Age', 'HowTallInMeters', 'Weight', 'BodyMassIndex', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4', 'Class']]= minmax_scale(harDataset_df[['User','Gender', 'Age', 'HowTallInMeters', 'Weight', 'BodyMassIndex', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4', 'Class']], feature_range=(0, 1)) 
###################################################################################################

#################################### STANDARDIZATION ##############################################
# # create a scaler object
std_scaler = StandardScaler()
# fit and transform the data
harDataset_df[['User','Gender', 'Age', 'HowTallInMeters', 'Weight', 'BodyMassIndex', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4', 'Class']] = pd.DataFrame(std_scaler.fit_transform(harDataset_df[['User','Gender', 'Age', 'HowTallInMeters', 'Weight', 'BodyMassIndex', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4', 'Class']]))
###################################################################################################

################################### CENTERING #####################################################
# harDataset_df[['User','Gender', 'Age', 'HowTallInMeters', 'Weight', 'BodyMassIndex', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4', 'Class']] = harDataset_df[['User','Gender', 'Age', 'HowTallInMeters', 'Weight', 'BodyMassIndex', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4', 'Class']].apply(lambda column: column -column.mean()).astype('float32')
###################################################################################################

x = harDataset_df.drop(['Class'], axis=1)
y = harDataset_df['Class']

# One-hot encode the output variable
y = np.array(y)
encoder = OneHotEncoder()
y = encoder.fit_transform(y.reshape(-1, 1)).toarray()

################################### K-FOLD CROSS VALIDATION #######################################
# We will use this 'kf'(StratiFiedKFold splitting stratergy) object as input to cross_val_score() method
# The folds are made by preserving the percentage of samples for each class.
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#####################################################################################################

# Split the data to training and testing data 5-Fold
kfold = KFold(n_splits=5, shuffle=True)
mseList = []
accuracyList = []
ceScoresList = []

# define early stopping criterion
es = EarlyStopping(monitor='loss', patience=10)

for i, (train, test) in enumerate(kfold.split(x)):

       x_train, x_test, y_train, y_test = x.iloc[train], x.iloc[test], y[train], y[test]
       
       # Define the model
       model = Sequential()
       model.add(Dense(23, input_dim=18, activation='relu', kernel_regularizer=l2(0.9)))
       model.add(Dense(5, activation='softmax'))

       # Compile the model
       # model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.1, momentum=0.6), metrics=['categorical_crossentropy', 'accuracy'])
       model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.1, momentum=0.6), metrics=['mean_squared_error', 'accuracy'])
       
       # Fit the model for this fold
       history = model.fit(x_train, y_train, epochs=10, batch_size=32, callbacks=[es], verbose=2)

       # Evaluate model
       scores = model.evaluate(x_test, y_test, verbose=0)
       mseList.append(scores[1])
       accuracyList.append(scores[2])
       # ceScoresList.append(scores[1])

       print("Scores", scores, "Fold :", i, " MSE:", scores[1], "Accuracy:", scores[2])
       # print("Scores", scores, "Fold :", i, " crossEntropyScores:", scores[1], "Accuracy:", scores[2])

       # Plot categorical_crossentropy and accuracy for this fold
       # plt.plot(history.history['categorical_crossentropy'])
       # plt.plot(history.history['accuracy'])
       # plt.title(f'Model categorical_crossentropy - Fold Number {i}')
       # plt.ylabel('categorical_crossentropy')
       # plt.xlabel('Epoch')
       # plt.legend(['categorical_crossentropy', 'accuracy'], loc='center right')
       # plt.show()

       # Plot mean_squared_error and accuracy for this fold
       plt.plot(history.history['mean_squared_error'])
       plt.plot(history.history['accuracy'])
       plt.title(f'Model mean_squared_error - Fold Number {i}')
       plt.ylabel('mean_squared_error')
       plt.xlabel('Epoch')
       plt.legend(['mean_squared_error', 'accuracy'], loc='center right')
       plt.show()


print("MSE: ", np.mean(mseList), "Accuracy:", np.mean(accuracyList))
# print("crossEntropyScores: ", np.mean(ceScoresList), "Accuracy:", np.mean(accuracyList))

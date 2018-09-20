import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

data_cols = ['user id','movie id','rating','timestamp']

print(data_cols[2])
item_cols = ['movie id','movie_title','release date',
'video release date','IMDb URL','unknown','Action',
'Adventure','Animation','Childrens','Comedy','Crime',
'Documentary','Drama','Fantasy','FilmNoir','Horror',
'Musical','Mystery','Romance','SciFi','Thriller',
'War' ,'Western']

user_cols = ['user id','age','gender','occupation',
'zip code']


users = pd.read_csv('u.user', sep='|', names=user_cols, encoding='latin-1')
item = pd.read_csv('u.item', sep='|', names=item_cols, encoding='latin-1')
data = pd.read_csv('u.data', sep='\t', names=data_cols, encoding='latin-1')
dataset = pd.merge(pd.merge(item, data),users)
cols_at_end = ['rating']
dataset = dataset[[c for c in dataset if c not in cols_at_end] + [c for c in cols_at_end if c in dataset]]

dataset.drop(dataset.columns[[1,2,3,4,25,29]], inplace = True, axis = 1)


le = LabelEncoder()
for col in dataset.columns.values:
       if dataset[col].dtypes=='object':
           data = dataset[col]
           le.fit(data.values)
           dataset[col] = le.transform(dataset[col])
           print(dataset[col])


dataset.rating = dataset.rating.map({1:'Bad', 2:'Bad', 3:'Average', 4:'Good', 5:'Good'})
dataset.rating = dataset.rating.map({'Bad':0,'Bad':0,'Average':1,'Good':2,'Good':2})




x = dataset.iloc[:,0:24]
y = dataset.iloc[:,-1]

scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y, random_state = 0)

print("ACCURACY OF DIFFERENT CLASSIFIER'S FOR IMBALANCE DATASET")

model_1 = MLPClassifier(alpha = 100, random_state = 0).fit(x_train,y_train)
print("Neural Network Training Accuracy: {:.2f} %" .format(model_1.score(x_train, y_train)*100))
print("Neural Network Testing Accuracy: {:.2f} %" .format(model_1.score(x_test, y_test)*100))

model_2 = LogisticRegression(random_state = 0).fit(x_train,y_train)
print("Logistic Regression Training Accuracy: {:.2f} %" .format(model_2.score(x_train, y_train)*100))
print("Logistic Regression Testing Accuracy: {:.2f} %" .format(model_2.score(x_test, y_test)*100))

model_3 = RandomForestClassifier(n_estimators = 21, max_depth = 20, random_state = 0 ).fit(x_train,y_train)
print("Random Forest Training Accuracy: {:.2f} %" .format(model_3.score(x_train, y_train)*100))
print("Random Forest Testing Accuracy: {:.2f} %" .format(model_3.score(x_test, y_test)*100))


print("*****************************************************************")


dataset_majority = dataset[dataset.rating==2]
dataset_high_minority = dataset[dataset.rating==1]
dataset_minority = dataset[dataset.rating==0]

dataset_high_minority_upsampled = resample(dataset_high_minority, replace=True, n_samples=34174, random_state=123) 
dataset_minority_upsampled = resample(dataset_minority, replace=True, n_samples=34174, random_state=123) 
dataset_upsampled = pd.concat([dataset_high_minority_upsampled, dataset_minority_upsampled, dataset_majority])


x_upsample = dataset_upsampled.iloc[:,0:24]
y_upsample  = dataset_upsampled.iloc[:,-1]


scaler_upsample = StandardScaler()
scaler_upsample.fit(x_upsample)
x_scaled_upsample = scaler_upsample.transform(x_upsample)

x_train_upsampled, x_test_upsampled, y_train_upsampled, y_test_upsampled = train_test_split(x_scaled_upsample, y_upsample, random_state = 0)

print("ACCURACY OF DIFFERENT CLASSIFIER'S FOR UPSAMPLED DATASET")

upsampled_model_1 = MLPClassifier(random_state = 0).fit(x_train_upsampled,y_train_upsampled)
print("Neural Network Training Accuracy: {:.2f} %" .format(upsampled_model_1.score(x_train_upsampled, y_train_upsampled)*100))
print("Neural Network Testing Accuracy: {:.2f} %" .format(upsampled_model_1.score(x_test_upsampled, y_test_upsampled)*100))

upsampled_model_2 = LogisticRegression(random_state = 0).fit(x_train_upsampled,y_train_upsampled)
print("Logistic Regression Training Accuracy: {:.2f} %" .format(upsampled_model_2.score(x_train_upsampled, y_train_upsampled)*100))
print("Logistic Regression Testing Accuracy: {:.2f} %" .format(upsampled_model_2.score(x_test_upsampled, y_test_upsampled)*100))

upsampled_model_3 = RandomForestClassifier(n_estimators = 62, random_state = 0 ).fit(x_train_upsampled, y_train_upsampled)
print("Random Forest Training Accuracy: {:.2f} %" .format(upsampled_model_3.score(x_train_upsampled, y_train_upsampled)*100))
print("Random Forest Testing Accuracy: {:.2f} %" .format(upsampled_model_3.score(x_test_upsampled, y_test_upsampled)*100))


print("*****************************************************************")


dataset_high_majority = dataset[dataset.rating==2]
dataset_majority = dataset[dataset.rating==1]
dataset_minority = dataset[dataset.rating==0]

dataset_high_majority_downsampled = resample(dataset_high_majority, replace=False, n_samples=17480, random_state=123) 
dataset_majority_downsampled = resample(dataset_majority, replace=False, n_samples=17480, random_state=123) 

dataset_downsampled = pd.concat([dataset_high_majority_downsampled, dataset_majority_downsampled, dataset_minority])



x_downsample = dataset_downsampled.iloc[:,0:24]
y_downsample = dataset_downsampled.iloc[:,-1]


scaler_downsample = StandardScaler()
scaler_downsample.fit(x_downsample)
x_scaled_downsample = scaler_downsample.transform(x_downsample)

x_train_downsampled, x_test_downsampled, y_train_downsampled, y_test_downsampled = train_test_split(x_scaled_downsample, y_downsample, random_state = 0)

print("ACCURACY OF DIFFERENT CLASSIFIER'S FOR DOWNSAMPLED DATASET")

downsampled_model_1 = MLPClassifier(random_state = 0).fit(x_train_downsampled,y_train_downsampled)
print("Neural Network Training Accuracy: {:.2f} %" .format(downsampled_model_1.score(x_train_downsampled, y_train_downsampled)*100))
print("Neural Network Testing Accuracy: {:.2f} %" .format(downsampled_model_1.score(x_test_downsampled, y_test_downsampled)*100))

downsampled_model_2 = LogisticRegression(random_state = 0).fit(x_train_downsampled,y_train_downsampled)
print("Logistic Regression Training Accuracy: {:.2f} %" .format(downsampled_model_2.score(x_train_downsampled, y_train_downsampled)*100))
print("Logistic Regression Testing Accuracy: {:.2f} %" .format(downsampled_model_2.score(x_test_downsampled, y_test_downsampled)*100))

downsampled_model_3 = RandomForestClassifier(max_depth = 20,random_state = 0 ).fit(x_train_downsampled,y_train_downsampled)
print("Random Forest Training Accuracy: {:.2f} %" .format(downsampled_model_3.score(x_train_downsampled, y_train_downsampled)*100))
print("Random Forest Testing Accuracy: {:.2f} %" .format(downsampled_model_3.score(x_test_downsampled, y_test_downsampled)*100))


print("*****************************************************************")

predicted_1 = model_3.predict(x_test)
probabilities_1 = model_3.predict_proba(x_test)
print("\nClassification Report of Random Forest (Imbalanced Dataset: )")
print(classification_report(y_test, predicted_1, target_names=['Bad', 'Average', 'Good']))

predicted_2 = upsampled_model_3.predict(x_test_upsampled)
probabilities_2 = upsampled_model_3.predict_proba(x_test_upsampled)
print("\nClassification Report of Random Forest (Upsampled Dataset: )")
print(classification_report(y_test_upsampled, predicted_2, target_names=['Bad', 'Average', 'Good']))

print("Features by Importance (Random Forest, Upsampled Dataset): ")
n_features = x_upsample.shape[1]
plt.barh(range(n_features), upsampled_model_3.feature_importances_, align='center')
plt.yticks(np.arange(n_features), dataset.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.ylim(-1, n_features)

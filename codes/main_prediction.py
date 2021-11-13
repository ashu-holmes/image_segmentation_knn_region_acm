__author__ = 'dilip'
import numpy as np
import image
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2


def handle_non_numerical_data(df):
    # handling non-numerical data: must convert.
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        # print(column,df[column].dtype)
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:

            column_contents = df[column].values.tolist()
            # finding just the uniques
            unique_elements = set(column_contents)
            # great, found them.
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    # creating dict that contains new
                    # id per unique string
                    text_digit_vals[unique] = x
                    x += 1
            # now we map the new "id" vlaue
            # to replace the string.
            df[column] = list(map(convert_to_int, df[column]))

    return df


df = pd.read_csv('texture.txt')
df = handle_non_numerical_data(df)
x = np.array(df.drop(['class'],1))
y = np.array(df['class'])

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier(n_neighbors=8)
clf.fit(xtrain, ytrain)

accuracy = clf.score(xtest,ytest)
print ("accuracy")
print(accuracy)
example = cv2.imread("tiger.tif",cv2.IMREAD_GRAYSCALE)
example=example.reshape(512*256,1)
prediction = clf.predict(example)
print ("prediction")
print(prediction)
#u can give x instead of xtest
print(np.amax(clf.predict_proba(example), axis=1))
arr1=np.amax(clf.predict_proba(example), axis=1)
arr2=[0]*len(arr1)
np.set_printoptions(threshold=np.nan)
print ("arr1")
print(arr1)


for i in range(len(arr1)):
    if (arr1[i] < 0.25):
        arr2[i] = 0
    else:
        arr2[i] = 1

##print(clf.predict_proba(example))
##print ("prediction")
##print(prediction)


f = open('daret.txt','w')
f.write(str(arr2))
f.close()

arr3=[0]*len(arr1)
p=3
for i in range(len(arr1)):
    arr3[i]=(1+(2*arr1[i]-1)**p)/2

f = open('trutht.txt','w')
f.write(str(arr3))
f.close()

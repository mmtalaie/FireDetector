import cv2
import pandas
import numpy
from sklearn.svm import SVC,NuSVC,LinearSVC
from sklearn.model_selection import train_test_split
from PIL import Image

X=[]
label=[]
myDataSet=pandas.DataFrame()
for i in range(3):
    photo = cv2.imread("fire/fire"+str(i+1)+".png")
    photo = cv2.copyMakeBorder(photo, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    height, width, channels = photo.shape
    for y in range(1, height-1):
        for x in range(1,  width-1):
            f = (photo[y-1, x-1][0],photo[y-1, x-1][1],photo[y-1, x-1][2],
                 photo[y-1, x][0], photo[y-1, x][1], photo[y-1, x][2],
                 photo[y-1, x+1][0], photo[y-1, x+1][1], photo[y-1, x+1][2],
                 photo[y, x-1][0], photo[y, x-1][1], photo[y, x-1][2],
                 photo[y, x][0], photo[y, x][1], photo[y, x][2],
                 photo[y, x+1][0], photo[y, x+1][1], photo[y, x+1][2],
                 photo[y+1, x+1][0],photo[y+1, x+1][1],photo[y+1, x+1][2],
                 photo[y+1, x+1][0], photo[y+1, x+1][1], photo[y+1, x+1][2],
                 photo[y+1, x+1][0], photo[y+1, x+1][1], photo[y+1, x+1][2])
            X.append(f)
            label.append(1)

myDataSet = pandas.DataFrame(X, columns=["1", "2", "3",
                                         "4", "5", "6",
                                         "7", "8", "9",
                                         "10", "11", "12",
                                         "13", "14", "15",
                                         "16", "17", "18",
                                         "19", "20", "21",
                                         "22", "23", "24",
                                         "25", "26", "27"])
myDataSet["label"] = label
for i in range(3):
    photo = cv2.imread("fire/notfire"+str(i+1)+".png")
    photo = cv2.copyMakeBorder(photo, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    height, width, channels = photo.shape
    for y in range(1, height-1):
        for x in range(1, width-1):
            f = (photo[y-1, x-1][0],photo[y-1, x-1][1],photo[y-1, x-1][2],
                 photo[y-1, x][0], photo[y-1, x][1], photo[y-1, x][2],
                 photo[y-1, x+1][0], photo[y-1, x+1][1], photo[y-1, x+1][2],
                 photo[y, x-1][0], photo[y, x-1][1], photo[y, x-1][2],
                 photo[y, x][0], photo[y, x][1], photo[y, x][2],
                 photo[y, x+1][0], photo[y, x+1][1], photo[y, x+1][2],
                 photo[y+1, x+1][0],photo[y+1, x+1][1],photo[y+1, x+1][2],
                 photo[y+1, x+1][0], photo[y+1, x+1][1], photo[y+1, x+1][2],
                 photo[y+1, x+1][0], photo[y+1, x+1][1], photo[y+1, x+1][2])
            X.append(f)
            label.append(0)

myDataSet = pandas.DataFrame(X,  columns=["1", "2", "3",
                                         "4", "5", "6",
                                         "7", "8", "9",
                                         "10", "11", "12",
                                         "13", "14", "15",
                                         "16", "17", "18",
                                         "19", "20", "21",
                                         "22", "23", "24",
                                         "25", "26", "27"])
myDataSet["label"] = label


myDataSet = myDataSet.sample(frac=1).reset_index(drop=True)

Maindata=myDataSet.loc[:, '1':'27']
Mainlabel=myDataSet.loc[:, "label"]

def PrepairImage():
    test = []
    photo = cv2.imread("firetest.jpg")
    fret = photo
    photo = cv2.copyMakeBorder(photo, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    height, width, channels = photo.shape

    for y in range(1, height-1):
        for x in range(1, width-1):
            test.append((photo[y-1, x-1][0],photo[y-1, x-1][1],photo[y-1, x-1][2],
                 photo[y-1, x][0], photo[y-1, x][1], photo[y-1, x][2],
                 photo[y-1, x+1][0], photo[y-1, x+1][1], photo[y-1, x+1][2],
                 photo[y, x-1][0], photo[y, x-1][1], photo[y, x-1][2],
                 photo[y, x][0], photo[y, x][1], photo[y, x][2],
                 photo[y, x+1][0], photo[y, x+1][1], photo[y, x+1][2],
                 photo[y+1, x+1][0],photo[y+1, x+1][1],photo[y+1, x+1][2],
                 photo[y+1, x+1][0], photo[y+1, x+1][1], photo[y+1, x+1][2],
                 photo[y+1, x+1][0], photo[y+1, x+1][1], photo[y+1, x+1][2]))
    return test, width, height, fret

dataTrain,dataTest,labelTrain,labelTest=train_test_split(Maindata,Mainlabel,test_size=0.3,random_state=42)


cls = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
cls.fit(dataTrain,labelTrain)
print(cls.score(dataTest,labelTest))
test,sh0,sh1,photo=testImage()
result = cls.predict(test)

#img = Image.new("RGB",(sh1,sh0))
img = photo
#pixels = img.load()
height = img.shape[0]
width = img.shape[1]
for i in range(height):
    for j in range(width):
        if(result[(i * width)+j]==1):
            img[i,j] = (0,0,255)

cv2.imshow('image',img)
cv2.waitKey(0)


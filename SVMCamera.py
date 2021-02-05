import cv2
import pandas
import numpy
from sklearn.svm import SVC,NuSVC,LinearSVC
from sklearn.model_selection import train_test_split
from PIL import Image


def PrepairImage(photo):
    test = []
    fret = photo
    #photo = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)
    height, width, channels = photo.shape

    for y in range(0, height):
        for x in range(0, width):
            test.append(photo[y,x])
    return test, width, height,fret

X=[]
label=[]
myDataSet=pandas.DataFrame()
sizeOfDataSet =5
for i in range(sizeOfDataSet+1):
    photo = cv2.imread("dataset/fire"+str(i+1)+".png")
    #photo = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)
    height, width, channels = photo.shape
    for y in range(0, height):
        for x in range(0, width):
            f = photo[y,x][:]
            X.append(f)
            label.append(1)

myDataSet = pandas.DataFrame(X, columns=["R", "G", "B"])
myDataSet["label"] = label
for i in range(sizeOfDataSet+1):
    photo = cv2.imread("dataset/notfire"+str(i+1)+".png")
    #photo = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)
    height, width, channels = photo.shape
    for y in range(0, height):
        for x in range(0, width):
            X.append(photo[y,x])
            label.append(0)

myDataSet = pandas.DataFrame(X, columns=["R", "G", "B"])
myDataSet["label"] = label


myDataSet = myDataSet.sample(frac=1).reset_index(drop=True)

Maindata=myDataSet.loc[:, 'R':'B']
Mainlabel=myDataSet.loc[:, "label"]



dataTrain,dataTest,labelTrain,labelTest=train_test_split(Maindata,Mainlabel,test_size=0.3,random_state=42)


cls = SVC(C=0.8, cache_size=1000, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
cls.fit(dataTrain,labelTrain)
print(cls.score(dataTest,labelTest))
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (50, 50), interpolation = cv2.INTER_CUBIC)
    test,sh0,sh1,photo=PrepairImage(frame)
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
    img = cv2.resize(img, (200, 200), interpolation = cv2.INTER_CUBIC)
    cv2.imshow('image',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
cap.release()
cv2.destroyAllWindows()

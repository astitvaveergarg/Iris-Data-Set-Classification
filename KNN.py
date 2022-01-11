from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

dataset=datasets.load_iris()

features= dataset.data
FlowerClass=dataset.target

model=KNeighborsClassifier()
model.fit(features, FlowerClass)

Score=model.score(features, FlowerClass)
print("Accuracy: ", round(Score*100, 2), "Percent")

SepalLength=float(input("Enter Sepal Length (in cm): "))
SepalWidth=float(input("Enter Sepal Width (in cm): "))
PetalLength=float(input("Enter Petal Length (in cm): "))
PetalWidth=float(input("Enter Petal Width (in cm): "))

Prediction=model.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])

if (Prediction==0):
    print("Fower Class is Iris Setosa")
elif (Prediction==1):
    print("Fower Class is Iris Versicolour")
else:
    print("Fower Class is Iris Virginica")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("iris.csv")
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
Y = data['species']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .30, random_state = 1)

regr = LinearRegression()
regr.fit(X_train,Y_train)

y_pred = regr.predict(X_test).round()
accuracy =accuracy_score(Y_test,y_pred)
print("Model Accuracy: ", round(accuracy * 100, 4), "%")

print("\nEnter flower details:")
sl = float(input("sepal length: "))
sw = float(input("sepal width: "))
pl = float(input("petal length: "))
pw = float(input("petal width: "))

input_data = pd.DataFrame([[sl, sw, pl, pw]], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

new_pred = regr.predict(input_data).round()

if int(new_pred[0]) != 0 and int(new_pred[0]) != 1 and int(new_pred[0]) != 2:
    print("unknown species")

else:
    print("predicted species:", int(new_pred[0]))
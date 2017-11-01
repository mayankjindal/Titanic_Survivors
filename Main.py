import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

x = pd.read_csv("/home/mayank/Downloads/train.csv")
y = x.pop("Survived")
x["Age"].fillna(x.Age.mean(), inplace=True)
x.drop(["Name", "Ticket", "PassengerId"], axis=1, inplace=True)

def cabin(x):
    try:
        return x[0]
    except TypeError:
        return "None"

x["Cabin"] = x.Cabin.apply(cabin)

categorical_variables = ["Sex", "Cabin", "Embarked"]

for v in categorical_variables:
    x[v].fillna('Missing', inplace=True)
    dummies = pd.get_dummies(x[v], prefix=v)
    x = pd.concat([x, dummies], axis=1)
    x.drop([v], axis=1, inplace=True)

logreg = LogisticRegression()
logreg.fit(x, y)

k = pd.read_csv("/home/mayank/Downloads/test.csv")
k["Age"].fillna(k.Age.mean(), inplace=True)
k["Fare"].fillna(k.Fare.mean(), inplace=True)
k.drop(["Name", "Ticket", "PassengerId"], axis=1, inplace=True)

def cabin(k):
    try:
        return k[0]
    except TypeError:
        return "None"

k["Cabin"] = k.Cabin.apply(cabin)
categorical_variables = ["Sex", "Cabin", "Embarked"]

for v in categorical_variables:
    k[v].fillna('Missing', inplace=True)
    dummies = pd.get_dummies(k[v], prefix=v)
    k = pd.concat([k, dummies], axis=1)
    k.drop([v], axis=1, inplace=True)
k = pd.DataFrame(k, columns=x.keys())
k["Cabin_T"].fillna(0, inplace=True)
k["Embarked_Missing"].fillna(0, inplace=True)
z = logreg.predict(k)


k = pd.read_csv("/home/mayank/Downloads/test.csv")
index = list(range(0, len(k)))
k = pd.DataFrame(k, index=index)
data = {"PassengerId":[], "Survived":[]}
for i in range(0, len(k)):
    data["PassengerId"].append(k['PassengerId'][i])
    data["Survived"].append(z[i])

Tested = pd.DataFrame(data)
Tested.to_csv("/home/mayank/Downloads/Mayank_Jindal.csv", index=False)


#print("Accuracy : ", metrics.accuracy_score(y, z))

#correct, wrong = 0, 0
#for i in range(0, 891):
 #   if z[i] == y[i]  :
  #      correct += 1
   # else:
    #    wrong += 1
#print("Correctly guesses: ", correct, "Incorrect guesses: ",  wrong)
#print("Value counts: 1-Survived, 0-Died")
#print(pd.Series.value_counts(z))

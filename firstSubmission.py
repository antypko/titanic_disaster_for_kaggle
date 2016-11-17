import csv as csv
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier

input_train_ds = pd.read_csv('./train.csv', header=0)
input_test_ds = pd.read_csv('./test.csv', header=0)
###
# Columns that are taking into concider
valuable_values = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
# valuable_values = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

def prepare_titanic_data (input_data):
    input_data['Sex'] = input_data['Sex'].map({'female': 0, 'male': 1}).astype(int)
    input_data["Embarked"] = input_data["Embarked"].fillna('S')
    input_data['Embarked'] = input_data['Embarked'].map({'C': 3, 'Q': 2, 'S': 1}).astype(int)
    ####################
    # Filling age gaps
    train_median_ages = np.zeros((2, 3, 7))
    for x in range(0, 2):
        for y in range(0, 3):
            for z in range(0, 7):
                gender = input_data['Sex'] == x
                p_class = input_data['Pclass'] == y + 1
                sib_sp = input_data['SibSp'] == z
                train_median_ages[x, y, z] = input_data[gender & p_class & sib_sp]['Age'].median()

    print train_median_ages

    for x in range(0, 2):
        for y in range(0, 3):
            for z in range(0, 7):
                gender = input_data['Sex'] == x
                p_class = input_data['Pclass'] == y + 1
                sib_sp = input_data['SibSp'] == z
                age_is_null = input_data.Age.isnull()
                input_data.loc[age_is_null & gender & p_class & sib_sp, 'Age'] = train_median_ages[x, y, z]
    ##########################################################
    # Fill missing Fare(found only one example in test dataset)
    input_data["Age"] = input_data["Age"].fillna(input_data["Age"].median())
    input_data["Fare"] = input_data["Fare"].fillna(input_data["Fare"].median())

    return input_data

train_data = prepare_titanic_data(input_train_ds)
test_data = prepare_titanic_data(input_test_ds)

forest = RandomForestClassifier(n_estimators=100)

#"Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"
scores = cross_val_score(forest, train_data[valuable_values], train_data["Survived"], cv=3)
print(scores.mean())

forest.fit(train_data[valuable_values], train_data["Survived"])
predictions = forest.predict(test_data[valuable_values])

submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": predictions
})

submission.to_csv('random_forest_submission.csv', index=False)

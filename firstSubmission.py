import csv as csv
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

input_train_ds = pd.read_csv('./train.csv', header=0)
input_test_ds = pd.read_csv('./test.csv', header=0)
###
# Columns that are taking into concider
scalable_values = ["Age", "Fare"]
# valuable_values = ["First_Class", "Second_Class", "Female", "Child", "Family", "Fare", "C", "Q"]
valuable_values = ["First_Class", "Second_Class", "Third_Class", "Age", "Female", "Male", "Child", "Family", "Fare", "C", "Q", "S"]

def prepare_titanic_data (input_data):
    input_data["Embarked"] = input_data["Embarked"].fillna('S')
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

    sex_dummies = pd.get_dummies(input_data['Sex'])
    sex_dummies.columns = ['Female', 'Male']
    input_data = input_data.join(sex_dummies)

    embarked_dummies = pd.get_dummies(input_data['Embarked'])
    input_data = input_data.join(embarked_dummies)

    p_class_dummies = pd.get_dummies(input_data['Pclass'])
    p_class_dummies.columns = ["First_Class", "Second_Class", "Third_Class"]
    input_data = input_data.join(p_class_dummies)

    input_data['Family'] = input_data["Parch"] + input_data["SibSp"]
    input_data['Family'].loc[input_data['Family'] > 0] = 1
    input_data['Family'].loc[input_data['Family'] == 0] = 0

    input_data['Child'] = 0
    input_data['Child'].loc[input_data['Age'] <= 15] = 1

    scaler = StandardScaler()
    input_data[scalable_values] = scaler.fit_transform(input_data[scalable_values])

    return input_data

train_data = prepare_titanic_data(input_train_ds)
test_data = prepare_titanic_data(input_test_ds)
##
###############TRAINING###############
##


# C_range = [2, 2.5, 3, 3.5, 4]
# gamma_range = [0.025, 0.05, 0.06, 0.07, 0.075]
# results = []
# for C in C_range:
#     c_results = []
#     for gamma in gamma_range:
#         svm_check = SVC(kernel='rbf', C=C, gamma=gamma)
#         svm_score = cross_val_score(svm_check, train_data[valuable_values], train_data["Survived"], cv=10)
#         results.append((svm_score.mean(), C, gamma))
#         c_results.append(svm_score.mean())
#         # print 'svm_score: ', svm_score.mean(), ' for C:', C, ' gamma:', gamma
#     c_mean = np.array(c_results).mean()
#     print 'mean : ', c_mean, ' for C : ', C
#
#
# for gamma in gamma_range:
#     gamma_results = []
#     for C in C_range:
#         svm_check = SVC(kernel='rbf', C=C, gamma=gamma)
#         svm_score = cross_val_score(svm_check, train_data[valuable_values], train_data["Survived"], cv=10)
#         gamma_results.append(svm_score.mean())
#         gamma_mean = np.array(gamma_results).mean()
#     print 'mean : ', gamma_mean, ' for gamma : ', gamma



# logreg.fit(X_train, Y_train)
# Y_pred = logreg.predict(X_test)
# logreg.score(X_train, Y_train)
logreg = LogisticRegression()
forest = RandomForestClassifier(n_estimators=100)
# k_neighbors = KNeighborsClassifier(2)
svm_rbf = SVC(kernel='rbf', C=3.5, gamma=0.06)
# svm_sigmoid = SVC(kernel='sigmoid')

forest.fit(train_data[valuable_values], train_data["Survived"])
forest_cros_scores = cross_val_score(forest, train_data[valuable_values], train_data["Survived"], cv=10)
forest_score = forest.score(train_data[valuable_values], train_data["Survived"])
print 'forest_scores: ', forest_cros_scores.mean(), " and: ", forest_score
# scores_svm_linear = cross_val_score(svm_linear, train_data[valuable_values], train_data["Survived"], cv=3)
# print 'scores_svm_linear: ', scores_svm_linear.mean()
svm_rbf_cros_scores = cross_val_score(svm_rbf, train_data[valuable_values], train_data["Survived"], cv=10)
svm_rbf.fit(train_data[valuable_values], train_data["Survived"])
svm_rbf_score = svm_rbf.score(train_data[valuable_values], train_data["Survived"])
print 'scores_svm_rbf: ', svm_rbf_cros_scores.mean(), " and: ", svm_rbf_score
logreg_cros_scores = cross_val_score(logreg, train_data[valuable_values], train_data["Survived"], cv=10)
logreg.fit(train_data[valuable_values], train_data["Survived"])
logreg_score = logreg.score(train_data[valuable_values], train_data["Survived"])
print 'scores_logreg: ', logreg_cros_scores.mean(), " and: ", logreg_score


# scores_svm_sigmoid = cross_val_score(svm_sigmoid, train_data[valuable_values], train_data["Survived"], cv=3)
# print 'scores_svm_sigmoid: ', scores_svm_sigmoid.mean()
# scores_k_neighbors = cross_val_score(k_neighbors, train_data[valuable_values], train_data["Survived"], cv=10)
# print 'scores_k_neighbors: ', scores_k_neighbors.mean()
# print(forest_scores.mean())


# forest.fit(train_data[valuable_values], train_data["Survived"])
random_forest_predictions = forest.predict(test_data[valuable_values])

# svm_rbf.fit(train_data[valuable_values], train_data["Survived"])
svm_rbf_predictions = svm_rbf.predict(test_data[valuable_values])
#
# svm_linear.fit(train_data[valuable_values], train_data["Survived"])
# svm_linear_predictions = svm_linear.predict(test_data[valuable_values])

random_forest_submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": random_forest_predictions
})

svm_rbf_submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": svm_rbf_predictions
})
#
svm_rbf_submission.to_csv('svm_rbf_submission.csv', index=False)
random_forest_submission.to_csv('random_forest_submission.csv', index=False)
#
# svm_linear_submission = pd.DataFrame({
#     "PassengerId": test_data["PassengerId"],
#     "Survived": svm_linear_predictions
# })
#
# svm_linear_submission.to_csv('svm_linear_submission.csv', index=False)

# submission.to_csv('random_forest_submission.csv', index=False)

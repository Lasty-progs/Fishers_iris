import csv
from random import random

# Open file part
file = open("iris.csv", 'r') # sepal_length,sepal_width,petal_length,petal_width,species
csv_reader = csv.reader(file)
dataset = []

for row in csv_reader:
    
    species = {
        "Iris-setosa":[1,0,0],
        "Iris-versicolor":[0,1,0],
        "Iris-virginica":[0,0,1]
    }

    dataset.append(([float(row[0]),float(row[1]),float(row[2]),float(row[3])],species[row[4]]))

file.close()
file.close()

# Dataset opened and prepaared for operations

def predict(input, weight):
    out = [0,0,0]
    for i in range(3):
        for j in range(4):
            out[i] += input[j] * weight[i][j]
    return out

def rebalance(input, weight, pred, goal_pred):
    out_weight = []
    for i in range(len(weight)):
        alpha = 0.01
        out_weight.append([])
        for j in range(len(weight[0])):
            out_weight[i].append(weight[i][j] - input[j] * (pred[i] - goal_pred[i]) * alpha)

    return out_weight

def calc_out(pred):
    mx = max(pred)
    out = [0,0,0]
    for i in range(len(pred)):
        if pred[i] == mx:
            out[i] = 1
    return out

# Creating neuron web with 4 inputs and 3 outputs

input = []
goal_pred = []
weight = []
epochs = 10

for i in range(3): # Add matrix random weights
    weight.append([random(),random(),random(),random()])

# Train web
for i in range(epochs):
    for j in range(len(dataset)):
        print("Epoch: " + str(i) + " Instance: " + str(j))
        input = dataset[j][0]
        goal_pred = dataset[j][1]
        pred = predict(input, weight)
        
        errors = [(pred[x] - goal_pred[x]) ** 2 for x in range(3)]

        print("Errors: " + str(errors))

        weight = rebalance(input, weight, pred, goal_pred)


# Test cases

print("------Test Cases------")

# 4.3,3,1.1,0.1,Iris-setosa
# 5.8,4,1.2,0.2,Iris-setosa
# 5.7,4.4,1.5,0.4,Iris-setosa

# 6.1,3,4.6,1.4,Iris-versicolor
# 5.8,2.6,4,1.2,Iris-versicolor
# 5,2.3,3.3,1,Iris-versicolor

# 7.9,3.8,6.4,2,Iris-virginica
# 6.4,2.8,5.6,2.2,Iris-virginica
# 6.3,2.8,5.1,1.5,Iris-virginica

test = [
[[4.3, 3, 1.1, 0.1], [1,0,0]],
[[5.8,4,1.2,0.2], [1,0,0]],
[[5.7,4.4,1.5,0.4], [1,0,0]],
[[6.1,3,4.6,1.4], [0,1,0]],
[[5.8,2.6,4,1.2], [0,1,0]],
[[5,2.3,3.3,1], [0,1,0]],
[[7.9,3.8,6.4,2], [0,0,1]],
[[6.4,2.8,5.6,2.2], [0,0,1]],
[[6.3,2.8,5.1,1.5], [0,0,1]]
]

for i in range(len(test)):
    input = dataset[i][0]
    goal_pred = dataset[i][1]
    pred = predict(input, weight)

    errors = [(pred[x] - goal_pred[x]) ** 2 for x in range(3)]
    pred = calc_out(pred)
    if pred == goal_pred:
        print("Complete")
    else:
        print("Error")

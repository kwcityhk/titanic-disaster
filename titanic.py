"""
    Before using this script you have to download test.csv and train.csv from this site: https://www.kaggle.com/c/titanic/data
"""

import csv as csv
import numpy as np

# load in the csv file and read it as text
csv_file_object = csv.reader(open('train.csv', 'rU'))
# we want to store the csv data inside a numpy array
data=np.array(csv_file_object.next())
data=np.expand_dims(data,axis=0)

# save each row to our new data array
for row in csv_file_object:
    data = np.append(data,[row],axis=0)

header = data[0]
data = np.delete(data,0,0)
print("header: ",header)
hd = {}
i = 0
for h in header:
    hd[h] = i
    i += 1

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

"""
    Predicts the outcome for the input x and the weights w
    x_0 is 1 and w_0 is the bias
"""
def predict(inp,weights):
    return sigmoid(np.dot(inp.T,weights.T))

"""
    get the gradient for the inputs x,weights w and the outputs y
"""
def gradient(x,w,y):
    # create an empty gradient array which has the same length as the array of weights
    grads = np.empty((len(w)))
    # compute the gradient for each weight
    for j in range(len(w)):
        grads[j] = np.mean(np.sum([x[i,j]*(predict(x[i],w)-y[i]) for i in range(len(x))]))
    return grads

"""
    get the new weights based on the old ones w, learning rate a and the gradients
"""
def getWeights(w,a,grads):
    return w-a*grads

"""
    Determine the cost of the prediction (pred)
"""
def cost(real,pred):
    return np.sqrt(1.0/2*np.mean(np.power(real-pred,2)))

def getInputRepresentation(m,entry,test=False):
    if test:
        t = 1
    else:
        t = 0

    inp = np.zeros(m+1)
    inp[0] = 1
    inp[1] = entry[hd['Sex']-t] == "female"
    pClass = entry[hd['Pclass']-t].astype(np.float)
    inp[2] = pClass == 1
    inp[3] = pClass == 2
    inp[4] = pClass == 3

    # we can also add the age column and divide it into 5 groups
    if entry[hd['Age']-t] == '':
        inp[5:10] = [0.1,0.3,0.2,0.2,0.2]
    else:
        inp[5] = (entry[hd['Age']-t].astype(np.float) <= 15)
        inp[6] = (entry[hd['Age']-t].astype(np.float) > 15) & (entry[hd['Age']-t].astype(np.float) <= 25)
        inp[7] = (entry[hd['Age']-t].astype(np.float) > 25) & (entry[hd['Age']-t].astype(np.float) <= 32)
        inp[8] = (entry[hd['Age']-t].astype(np.float) > 32) & (entry[hd['Age']-t].astype(np.float) <= 41)
        inp[9] = (entry[hd['Age']-t].astype(np.float) > 41)


    inp[10] = (entry[hd['SibSp']-t].astype(np.float) <= 0)
    inp[11] = (entry[hd['SibSp']-t].astype(np.float) > 0) & (entry[hd['SibSp']-t].astype(np.float) <= 1)
    inp[12] = (entry[hd['SibSp']-t].astype(np.float) > 1)
    inp[13] = (entry[hd['Parch']-t].astype(np.float) <= 0)
    inp[14] = (entry[hd['Parch']-t].astype(np.float) > 0) & (entry[hd['Parch']-t].astype(np.float) <= 1)
    inp[15] = (entry[hd['Parch']-t].astype(np.float) > 1)
    if entry[hd['Fare']-t] == '':
        inp[16:21] = [0.2,0.2,0.2,0.2,0.2]
    else:
        inp[16] = (entry[hd['Fare']-t].astype(np.float) <= 8)
        inp[17] = (entry[hd['Fare']-t].astype(np.float) > 8) & (entry[hd['Fare']-t].astype(np.float) <= 11)
        inp[18] = (entry[hd['Fare']-t].astype(np.float) > 11) & (entry[hd['Fare']-t].astype(np.float) <= 22)
        inp[19] = (entry[hd['Fare']-t].astype(np.float) > 22) & (entry[hd['Fare']-t].astype(np.float) <= 40)
        inp[20] = (entry[hd['Fare']-t].astype(np.float) > 40)

    title = entry[hd['Name']-t].split(", ")[1].split(" ")[0]
    inp[21] = title == "Mr."
    inp[22] = title == "Mrs."
    inp[23] = title == "Miss."
    inp[24] = title == "Master."
    inp[25] = title == "Dr."
    inp[26] = title == "Sir."
    if (np.count_nonzero(inp[21:27]) == 0):
        inp[27] = 1
    return inp


inputs = []
outputs = []
m = 27 # number of features without threshold

for entry in data:
    inp = getInputRepresentation(m,entry)

    inputs.append(inp)
    outputs.append(entry[hd['Survived']])

inputs = np.array(inputs).astype(np.float)
outputs = np.array(outputs).astype(np.float)

weights = np.random.rand(m+1) # one for the threshold
alpha = 0.001
epochs = 100
train_size = int((3*len(inputs))/4)
trainX = inputs[0:train_size]
trainY = outputs[0:train_size]
testX = inputs[train_size:]
testY = outputs[train_size:]

for t in range(epochs):
    weights = getWeights(weights,alpha,gradient(trainX,weights,trainY))
    sum_costs = 0
    for inp,outp in zip(testX,testY):
        prediction = predict(inp,weights)
        last_cost = cost(outp,0 if prediction < 0.5 else 1)
        sum_costs += last_cost

    print(weights)
    print(sum_costs/(len(inputs)-train_size))


# First, read in test.csv
test_file = open('test.csv', 'rU')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

# Write out the PassengerId, and my prediction.
predictions_file = open("prediction.csv", "w")
predictions_file_object = csv.writer(predictions_file)
# write the column headers
predictions_file_object.writerow(["PassengerId", "Survived"])
# For each row in test file,
for row in test_file_object:
    inp = getInputRepresentation(m,np.array(row),test=True)
    prediction = predict(inp,weights)
    predictions_file_object.writerow([row[0], "0" if prediction < 0.5 else "1"])

# Close out the files.
test_file.close()
predictions_file.close()

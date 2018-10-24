import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import numpy as np
import string
import random
import os
import pickle

#NETWORK IDEA
#Image CNN that outputs distribution and Text that outputs distribution that then goes into a fully connected network.

globalLetters = string.ascii_letters + " "
globalLettersLen = len(globalLetters)
validLabels = ['Food', 'Health and Beauty Care', 'Cleaning, Housekeeping, Paper', 'Baby', 'Pet', 'Kitchen', 'Home Decor, Window Treatments, Storage & Appliances', 'Bedding & Comforters', 'Sheets, Pillows, Towels, Mats', 'Area Rugs', 'Furniture', 'Mattresses', 'Sports, Outdoor, Leisure, Hardware', 'Electronics', 'Toys', 'Unknown - Operators Discretion', 'Office Supplies', 'Book', 'Video Game', 'Movie', 'Music', 'Apparel - Accessories', 'Apparel - Shoes', 'Apparel - Mens - Tops/Sleepwear/Uncategorized', 'Apparel - Mens - Bottoms/Underwear/Swimwear', 'Apparel - Womens - Tops/Sleepwear/Uncategorized', 'Apparel - Womens - Bottoms/Underwear/Swimwear', 'Apparel - Kids - Tops/Sleepwear/Uncategorized & Infant', 'Apparel - Kids - Bottoms/Underwear/Swimwear', 'Apparel - All Outerwear', 'Seasonal - Lawn & Garden', 'Seasonal - Spring/Summer', 'Seasonal - Halloween', 'Seasonal - Christmas', 'Seasonal - Winter']
numOfLabels = len(validLabels)
word_dict = {}
word_weights = None
dir_path = os.path.dirname(os.path.realpath(__file__))

def getData():
    dataList = []
    with open(dir_path+'/data.csv', newline='', encoding='utf-8') as csvData:
        reader = csv.reader(csvData, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        for row in reader:
            rowStr = ", ".join(row[:-1])
            label = row[-1]
            dataList.append([rowStr, label])
    for item in dataList:
        for char in item[0]:
            if (checkIfValidChar(char)) != True :
                item[0] = item[0].replace(char, " ")
    for item in dataList:
        item[0] = item[0].split(" ")
    for item in dataList:
        item[0] = filterSpaces(item[0])
    dataList = filterValidClass(dataList)
    for item in dataList:
        item[1] = getClassNum(item[1])
    random.shuffle(dataList)
    return dataList

def filterSpaces(array):
    temp = []
    for item in array:
        if item != "":
            temp.append(item)
    return temp

def filterValidClass(dataList):
    temp = []
    for item in dataList:
        for label in validLabels:
            if (item[1] == label):
                temp.append(item)
                break
    return temp

def getClassNum(name):
    return validLabels.index(name)

def splitDescription(dataList):
    for item in dataList:
        item[0] = item[0].split()
    return dataList

def checkIfValidChar(char):
    if (globalLetters.find(char) == -1):
        return False
    else:
        return True

def genVocab(dataList):
    gloveIndexFile = open(dir_path+'/gloveDictionary_index.pkl', 'rb')
    gloveWeightsFile = open(dir_path+'/gloveDictionary_weights.pkl', 'rb')
    global word_dict
    global word_weights
    word_dict = pickle.load(gloveIndexFile)
    word_weights = pickle.load(gloveWeightsFile)
   # weightSize = temp.size()[0]
    #weightAdd = 0
    #for item in dataList:
    #    for word in item[0]:
    #        if word not in word_dict:
    #            if (len(word_dict) > 0):
    #                word_dict[word] = len(word_dict)
    #                weightAdd += 1
    #            else:
    #                word_dict[word] = 0
    #                weightAdd += 1
    #toAdd = torch.normal(mean=0.5, std=torch.rand(weightAdd, 50)).float()
    #print(toAdd)
    #PLACEHOLDER
    #word_weights = torch.cat([temp.float(), toAdd])
    #word_weights = temp
    #print(word_weights)

class CNN(nn.Module):
    def __init__(self, vocabLen, embeddingDim, numOfConvolutions):
        super(CNN, self).__init__()
        self.embeddings = nn.Embedding(vocabLen, embeddingDim)
        self.embeddings.weight = nn.Parameter(word_weights)
        self.conv1h = nn.Conv2d(1, 20, (1, embeddingDim + 16), padding=8)
        self.conv2h = nn.Conv2d(1, 20, (2, embeddingDim + 16), padding=8)
        self.conv3h = nn.Conv2d(1, 20, (3, embeddingDim + 16), padding=8)
        self.conv4h = nn.Conv2d(1, 20, (4, embeddingDim + 16), padding=8)
        self.conv5h = nn.Conv2d(1, 20, (5, embeddingDim + 16), padding=8)
        self.conv6h = nn.Conv2d(1, 20, (6, embeddingDim + 16), padding=8)
        self.conv7h = nn.Conv2d(1, 20, (7, embeddingDim + 16), padding=8)
        self.conv8h = nn.Conv2d(1, 20, (8, embeddingDim + 16), padding=8)
        self.poolConv = nn.Conv2d(20, 35, (20, 1))
        self.linear1 = nn.Linear(35, 35)
        self.linear2 = nn.Linear(35, 35)
        self.linear3 = nn.Linear(35, 35)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.numOfConvolutions = numOfConvolutions

    def pool(self, conv):
        poolHeight = conv.size()[2]
        maxPool = nn.MaxPool2d((poolHeight, 1))
        return maxPool(conv)
    
    def forward(self, X):
        q = self.embeddings(X).view((1, 1, -1, 50))
        pools = []
        for i in range(int(self.numOfConvolutions)):
            pools.append(self.pool(self.conv1h(q)))
            pools.append(self.pool(self.conv1h(q)))
            pools.append(self.pool(self.conv1h(q)))
            pools.append(self.pool(self.conv1h(q)))
            pools.append(self.pool(self.conv1h(q)))
            pools.append(self.pool(self.conv1h(q)))
            pools.append(self.pool(self.conv2h(q)))
            pools.append(self.pool(self.conv2h(q)))
            pools.append(self.pool(self.conv2h(q)))
            pools.append(self.pool(self.conv2h(q)))
            pools.append(self.pool(self.conv3h(q)))
            pools.append(self.pool(self.conv3h(q)))
            pools.append(self.pool(self.conv3h(q)))
            pools.append(self.pool(self.conv3h(q)))
            pools.append(self.pool(self.conv4h(q)))
            pools.append(self.pool(self.conv4h(q)))
            pools.append(self.pool(self.conv5h(q)))
            pools.append(self.pool(self.conv5h(q)))
            pools.append(self.pool(self.conv6h(q)))
            pools.append(self.pool(self.conv6h(q)))
        allMax = torch.cat(pools).view(1,20,-1,1)
        convedPools = self.poolConv(allMax)
        linTransform1 = self.linear1(convedPools.view(-1, 35))
        activation1 = F.leaky_relu(linTransform1)
        linTransform2 = self.linear2(activation1)
        activation2 = F.leaky_relu(linTransform2)
        linTransform3 = self.linear3(activation2)
        return self.logSoftmax(linTransform3)

def calculateMean(costList):
    sum = 0
    for item in costList:
        sum += item
    return sum / len(costList)

def getGuess(o):
    val, index = o[0].max(0)
    return validLabels[index]

def getTop5(output):
    outputList = []
    outputExp = output[0].exp()
    outputSum = outputExp.sum()
    print(outputExp)
    for i in range(len(output[0])):
        outputList.append([validLabels[i], (outputExp[i]/outputSum)*100,'%'])
    return outputList

def getLabel(tensor):
    npArr = tensor.numpy()
    val = npArr[0]
    return validLabels[val]

def getModelAccuracy(model, X, Y):
    count = 0
    right = 0
    total = 0
    for item in X:
        output = model.forward(item)
        if (getGuess(output) == getLabel(Y[count])):
            right += 1
        count += 1
        total += 1
    return right/total

def getWords(text):
    toReturn = ""
    for char in text:
        if char in globalLetters:
            toReturn += char
    toReturn = toReturn.lower()
    return toReturn

def getValidWords(textArr):
    inputArr = []
    for word in textArr:
        newWord = getWords(word)
        if newWord in word_dict:
            inputArr.append(newWord)
    return filterSpaces(inputArr)

def filterEmpty(dataList):
    newList = []
    for item in dataList:
        if (len(item[0]) > 0):
            newList.append(item)
    return newList

def main(epochs):
    print("Loading data...")    
    dataList = getData()
    dataList = dataList[0:-3000]
    testList = dataList[-3000:]
    print("Loading vocab...")
    genVocab(dataList)
    print("dataList Length: ", len(dataList))
    print("testList Length: ", len(testList))
    print("word_dict length: ", len(word_dict))
    print("word_weights size: ", word_weights.size())
    cnn = CNN(len(word_dict), 50, 1)
    inputList = []
    categoryList = []
    testInputList = []
    testCategoryList = []
    for i in range(len(dataList)):
        dataList[i][0] = getValidWords(dataList[i][0])
    dataList = filterEmpty(dataList)
    for i in range(len(testList)):
        testList[i][0] = getValidWords(testList[i][0])
    testList = filterEmpty(testList)
    for item in dataList:
            inputList.append(torch.tensor([word_dict[word] for word in getValidWords(item[0])], dtype=torch.long))
            categoryList.append(torch.tensor(item[1], dtype=torch.long).view(1))
    for item in testList:
        testInputList.append(torch.tensor([word_dict[word] for word in getValidWords(item[0])], dtype=torch.long))
        testCategoryList.append(torch.tensor(item[1], dtype=torch.long).view(1))
    optimizer = optim.Adam(cnn.parameters(), lr=0.000007)
    cost_function = nn.NLLLoss()
    costList = []
    for epoch in range(epochs):
        itemCount = 0
        epochAvg = []
        for item in inputList:
            output = cnn.forward(item)
            cost = cost_function(output, categoryList[itemCount])
            costList.append(cost)
            epochAvg.append(cost)
            cost.backward()
            optimizer.step()
            if((itemCount % 100) == 0):
                    print("Cost after iteration " + str(itemCount) + ":", cost)
                    print("Avg cost after iteration " + str(itemCount) + ":", calculateMean(costList), "EPOCH AVG: ", calculateMean(epochAvg), "EPOCH", str(epoch) + "/" + str(epochs))
                    print("Label: ", getLabel(categoryList[itemCount]))
                    print("Guess: ", getGuess(output))
            itemCount += 1
    accuracy = getModelAccuracy(cnn, inputList, categoryList)
    print("Training set accuracy: ", (accuracy * 100), "%")
    testAccuracy = getModelAccuracy(cnn, testInputList, testCategoryList)
    print("Test set accuracy: ", (testAccuracy * 100), "%")
    torch.save(cnn, dir_path + '/cnn_model')

def initWordDict():
    #dataList = getData()
    dataList = []
    genVocab(dataList)

def loadModel():
    print("Loading CNN...")
    initWordDict()
    cnn = torch.load(dir_path + '/cnn_model.txt')
    for i in range(100):
        text = input("Enter text: ")
        text = getWords(text)
        print("New text: ", text)
        print("--------START--------")
        textArr = text.split(" ")
        x = 0
        inputArr = []
        try:
            for word in textArr:
                if word in word_dict:
                    inputArr.append(word)
                else:
                    print("Not in vocab: ", word)
            x = torch.tensor([word_dict[word] for word in inputArr], dtype=torch.long)
        except:
            print("ERROR")
        output = cnn.forward(x)
        percentageList = getTop5(output)
        percentageList.sort(key=lambda x: float(x[1]))
        percentageList.reverse()
        print("----------------")
        print("CNN guess: ", getGuess(output))
        print("----------------")
        for item in percentageList:
            print(item[0], round(float(item[1]) * 10000) / 10000, item[2])
        print("--------END--------")

def init():
    choose = input("To train this CNN enter 0. To load a saved CNN press 1.")
    if (choose == "0"):
        main(5)
    else:
        loadModel()

init()
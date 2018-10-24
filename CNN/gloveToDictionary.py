import os
import torch
import pickle

def transformIntoTensor(numString):
    numList = numString.split(" ")
    numListLen = len(numList)
    for i in range(numListLen):
        numList[i] = float(numList[i])
    tensor = torch.tensor(numList, dtype=torch.double)
    return tensor
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
gloveRawFile = open(dir_path + '/glove.6B.50d.txt').read()
gloveList = gloveRawFile.split("\n")[0:-1]
print("Glove Raw File Length: ", len(gloveList))
gloveDictionary = {}
gloveIndex = {}
print("Converting text file to dictionary...")
indexCounter = 0
for item in gloveList:
    spaceIndex = 0
    for char in item:
        if (char == " "):
            gloveDictionary[item[0:spaceIndex]] = transformIntoTensor(item[(spaceIndex+1):])
            gloveIndex[item[0:spaceIndex]] = indexCounter
            break
        spaceIndex += 1
    indexCounter += 1
gloveMatrix = torch.zeros(len(gloveIndex), 50, dtype=torch.double)
matrixCounter = 0
for key in gloveDictionary:
    gloveMatrix[matrixCounter] = gloveDictionary[key]
    matrixCounter += 1
print("Glove Dictionary Length: ", len(gloveDictionary))
print("Glove Index Length: ", len(gloveIndex))
print("Glove Matrix Size: ", gloveMatrix.size())
print("Writing Glove to files...")
gloveDictionaryFile = open(dir_path + "/gloveDictionary_dict.pkl", 'wb')
gloveIndexFile = open(dir_path + "/gloveDictionary_index.pkl", 'wb')
gloveWeightsMatrixFile = open(dir_path+"/gloveDictionary_weights.pkl", "wb")
pickle.dump(gloveDictionary, gloveDictionaryFile)
pickle.dump(gloveIndex, gloveIndexFile)
pickle.dump(gloveMatrix, gloveWeightsMatrixFile)
print("Done creating Glove Dictionary Files")


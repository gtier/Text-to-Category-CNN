# Text-to-Category-CNN

## Starting the neural network
This python script requires that python and Pytorch(https://pytorch.org/) be installed on the host system.
To start running the neural network script, enter the following command into the terminal: 
```
$ python (path to the TextClassifierCNN.py file)
```
You will be greeted by the following text: “To train this CNN enter 0. To load a saved CNN press 1.” Respond by entering either 0 or 1 into the terminal. Instructions regarding how to train this CNN are listed under the **Training this CNN** section. On the other hand, instructions on loading a saved model are in the **Loading a saved model** section.

## Training this CNN
**Note:** *The data used to train this neural network is not included in this repo. The code above is meant for public viewing, but the data that was used to train this network is not. However, you can use this model on a variety of natural language processing problems. To do so, you will need to follow the instructions bellow*.

The python script will only read and process a training set that is called “data.csv”.
After a model is trained, it will be saved as a file called “cnn_model”. Because there already exists a “cnn_model” file, please make sure to rename the currently saved model to something other than “cnn_model” before you train a new model (if you don’t, the file will be overwritten).

## Loading a saved model
The python script, when directed to load a saved CNN, will retrieve the neural network model from a file named “cnn_model”. Thus, if you have trained multiple models, make sure to name the one you want to load “cnn_model”.
If you have not modified the python script, you will be prompted for a text input, which the CNN will then use as its input.
If you wish to pass data into the CNN in another way, you must edit the `loadModel()` function. If you wish to pass in text from another source, simply assign said text to the **Text** variable on line 248.
**(IMPORTANT)** There is a **for** loop on line 247 directly above the **Text** variable on line 248, which dictates the number of times you can pass information into the CNN and receive a response without restarting the program. If you wish to make it run forever until a condition is satisfied, change line 247 to include your own custom logic.

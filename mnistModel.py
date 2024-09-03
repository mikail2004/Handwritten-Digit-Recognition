import keras
import numpy as np
from matplotlib import pyplot as plt

#--------------- Setting Up Data ----------------
kd = keras.datasets
modelDataSet = kd.mnist.load_data()
images, labels = modelDataSet

#Divide by 255 for normalization. In the MNIST dataset, pixel values range from 0 to 255. 
#These values represent the intensity of each pixel in the grayscale image:
#0: Black (no intensity)
#255: White (full intensity)

'''
Normalization: The pixel values in the MNIST dataset range from 0 to 255. 
To bring these values into the range [0, 1], we divide by 255. 
This is a common preprocessing step called normalization, 
which helps improve the convergence of the neural network training.

Reshaping: The reshape method is used to flatten the 28x28 images into 1D arrays of size 784 (since 28*28 = 784). 
This is necessary because the neural network expects a 1D input vector rather than a 2D image.
The -1 in reshape(-1, 784) automatically infers the correct size for the first dimension 
based on the total number of elements and the specified size for the second dimension.

One-hot: The labels in the MNIST dataset are integers ranging from 0 to 9. 
One-hot encoding converts these integer labels into a binary matrix representation. 
For example, the label 2 is converted to [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]. 
This is done using np.eye(10)[y_train], where np.eye(10) creates a 10x10 identity matrix, 
and indexing it with the label values creates the one-hot encoded vectors.
'''

#60,000 training images and 10,000 testing images of handwritten digits, each of size 28x28 pixels.
#Keras MNIST by default is a tuple containing 2 arrays (train and test). Hence no choice but to do this:
#x_train = 2D vector of the pixel values of an image. y_train = vector of labels.
(x_train, y_train), (x_test, y_test) = kd.mnist.load_data() 
x_train = x_train.reshape(-1, 784) / 255.0 
x_test = x_test.reshape(-1, 784) / 255.0

# One-hot encode the labels
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

weightsInputToHidden = np.random.uniform(-0.5, 0.5, (20, 784))
weightsHiddenToOutput = np.random.uniform(-0.5, 0.5, (10, 20))

biasWeightsInputToHidden = np.zeros((20, 1))
biasWeightsHiddenToOutput = np.zeros((10, 1))

'''
zip():
1. Iterates through multiple tuples to add them together as a new tuple.
2. The values of the first index of tuple 1 is paired with the value at the first index of tuple 2. 
3. And so on for each successive index.
'''

nrCorrect = 0
learnRate = 0.01
epochs = 3
for epoch in range(epochs):
    # img: 1D vector of 784 elements (flattened image)
    # l: 1D vector of 10 elements (one-hot encoded label)
    for img, l in zip(x_train, y_train):
        img = img.reshape(-1, 1) # Reshaping the flattened image into a column vector (784, 1)
        l = l.reshape(-1, 1) # Reshaping the one-hot encoded label into a column vector (10, 1)
        # Now img is a (784, 1) matrix and l is a (10, 1) matrix

        '''
        Reshaping the image into a (784, 1) matrix allows it to be multiplied with weight matrices. 
        For example, if the weight matrix between the input layer and the first hidden layer has a shape of (N, 784) 
        (where N is the number of neurons in the hidden layer), the matrix multiplication is possible.
        The result of this multiplication is a (N, 1) matrix, which matches the dimensionality required for further calculations.
        
        Consistency in Dimensions:
        Keeping inputs and labels in a consistent matrix format simplifies the implementation of the neural network.
        It ensures compatibility with the weight matrices and bias vectors, maintaining the mathematical operations' correctness.   
        '''

        #--------------- Training (Forward propagation) ----------------
        hiddenPreprocessing = biasWeightsInputToHidden + (weightsInputToHidden @ img)
        hiddenFinal = 1/(1 + np.exp(-hiddenPreprocessing)) 

        outputPreprocessing = biasWeightsHiddenToOutput + (weightsHiddenToOutput @ hiddenFinal)
        outputFinal = 1/(1 + np.exp(-outputPreprocessing))

        #--------------- Cost/Error Function ----------------
        e = 1/len(outputFinal) * np.sum((outputFinal-l) ** 2, axis=0) #Cost/Error Function
        nrCorrect += int(np.argmax(outputFinal) == np.argmax(l))
                
        #--------------- Backpropagation ---------------
        #1. Calculate the delta for each neuron (from the Output layer) => Cost Function Derivative
        deltaOutput = outputFinal - l #Normally this would be the derivative of the Cost/Error Function || But we use a trick
        weightsHiddenToOutput += -learnRate * deltaOutput @ np.transpose(hiddenFinal) #transpose: a matrix obtained from a given matrix by interchanging each row and the corresponding column -> I.E 3x2 matrix becomes 2x3 matrix.
        biasWeightsHiddenToOutput += -learnRate * deltaOutput 

        #2. Calculate the delta for each neuron (from the Hidden layer) => Activiation Function Derivative 
        deltaHidden = np.transpose(weightsHiddenToOutput) @ deltaOutput * (hiddenFinal * (1 - hiddenFinal))
        weightsInputToHidden += -learnRate * deltaHidden @ np.transpose(img) 
        biasWeightsInputToHidden += -learnRate * deltaHidden

    #Show accuracy for each epoch
    print(f"Epoch: {epoch+1}, Acc: {round((nrCorrect / x_train.shape[0]) * 100, 2)}%")
    nrCorrect = 0

#Show results
while True:
    index = int(input("Enter dataset index (0 - 9999) >>> "))
    img = x_test[index]

    plt.imshow(img.reshape(28, 28), cmap="Greys")
    img = img.reshape(-1, 1)

    #--------------- Running Model with Trained Weights ---------------
    #You can save the 4 weight values (biasWeightsInputToHidden, weightsInputToHidden, biasWeightsHiddenToOutput, weightsHiddenToOutput)
    # ^ To run the model externally without having to retrain and strain resources
    # Forward propagation input -> hidden
    h_pre = biasWeightsInputToHidden + weightsInputToHidden @ img
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden -> output
    o_pre = biasWeightsHiddenToOutput + weightsHiddenToOutput @ h
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(f"Model Prediction: {o.argmax()}")
    plt.show()
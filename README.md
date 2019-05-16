# Gomoku NN

A certain number of neural network take part in a tournament to create a AI able to play the game of Gomoku.</br>
The goal was to train a neural network to play Gomoku using the simplest methods we could find; here we chose a Multiperceptron neural network and neuro-evolution to train it.

## Neural Network

A simple Multiperceptron Neural Network uses feed forward method to predict the position to play. I wrote it from scratch (understand here that I didn't use any available machine learning framework such as Tensorflow or Pytorch but maybe I should have) using Numpy for Matrix manipulation.

It is train using a neuro-evolution method.

## Neuro-Evolution

Each individual neural network will compete against every other. After all matches are done, they all have a fitness score based thier win/lose ratio.

The best individuals are selected to "reproduce" (mix their weights into newly created neural network).

Every new individual will have some of their "genes" (weights) randomly modified to create more (sometimes less) fitted individuals whom we hope will perform better than their predecessors.

We repeat those steps for as needed epochs and VOILÀ!

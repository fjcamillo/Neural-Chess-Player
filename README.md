# Neural Chess Player < WIP >

----

In this project we will create a convolutional neural network that can recognize chess images from a camera angled from the top of the chess board then a fully connected neural network to predict the next move appropriate for the positions on the image

A Convolutional Neural Network best fits the project's goal. Using this Neural Network Architecture would allow us to recognize images seen by the camera setup. The basic idea behind a Convolotuional Neural Network, is to mimic how our brain breaks down the images captires by our eyes which is then abstracted by the gooey stuff in our brain.

----

# How will it work?

A camera will be placed on top of the gaming board making sure that it captures the whole board. Next, if the user is the white player he starts by moving a piece and then pressing enter on the keyboard. The press signifies the end of white's turn. The Neural Chess Player will then take an image and assesses the current position then predicts the next move. A terminal timer will act as the game clock. The Neural Chess Player will send a press call to the timer to say that the turn has ended.

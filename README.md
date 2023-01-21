# Computer Vision RPS
In this project a trained computer vision model was produced using [Teachable-Machine](https://teachablemachine.withgoogle.com/). Using this model, we can play a game of 'Rock, Paper, Scissors' by showing a hand gesture to the camera.

![](https://github.com/ScarfDemon/computer-vision-rock-paper-scissors/blob/main/image.jpg?raw=true)

## Training image project model using Teachable-Machine
Using [Teachable-Machine](https://teachablemachine.withgoogle.com/), four classes are made: Rock, Paper, Scissors, and Nothing. Each class is trained with images of me with the correct gesture (or no gesture for the Nothing class).

The model was then downloaded as a Tensorflow model as a folder containing the model as  `keras_model.h5`,  and labels `labels.txt`. These will be used in the check-model.py file when using our camera for the `user_choice` input when playing "Rock, Paper, Scissors".

## check-model.py
This file loads in the keras model and the camera captures the images of the user. Each image of the live video is normalised and processed such that the keras model can be applied to predict the class of the image being presented to it.


The variable `prediction` contains the output of the model, and each element in the output corresponds to the probability of the input image representing a particular class.

For example, if the prediction has the following output: [[0.8, 0.1, 0.05, 0.05]], there is an 80% chance that the input image shows rock, a 10% chance that it shows paper, a 5% chance that it shows scissors, and a 5% chance that it shows nothing.

## Requirements.txt file
To install the exact dependancies to run the "Rock, Paper, Scissors" Game, run `pip install requirements.txt`

## Rock Paper Scissors game methods

### `get_computer_choice()`
This function returns a random choice from 'Rock', 'Paper', 'Scissors'.

### `get_user_choice()`
This function asks the user for their choice.

### `get_winner(computer_choice, user_choice)`
This function takes two arguments: the computer's random choice, and the user's choice.

First the function checks the inputted choice by the user is valid. 

If the input is valid, the function then goes on to apply the rules of Rock, Paper, Scissors, letting the user know if they won, lost or tied.

### `play()`
This function runs the game, using the previously defined functions.


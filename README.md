# Computer Vision RPS
In this project a trained computer vision model was produced using [Teachable-Machine](https://teachablemachine.withgoogle.com/). Using this model, we can play a game of 'Rock, Paper, Scissors' by showing a hand gesture to the camera.

First to 3 points! If you win, a colourful message is displayed!

![](RPS-image.png)

## Manual Rock Paper Scissors
### `get_computer_choice()`
This function returns a random choice from 'Rock', 'Paper', 'Scissors'.

### `get_user_choice()`
This function asks the user for their choice.

### `get_winner(computer_choice, user_choice)`
This function takes two arguments: the computer's random choice, and the user's choice.

If the input is valid, the function then goes on to apply the rules of Rock, Paper, Scissors, letting the user know if they won, lost or tied.

### `play()`
This function runs the game, using the previously defined functions.

## Teachable-Machine keras model
Using [Teachable-Machine](https://teachablemachine.withgoogle.com/), four classes are made: Rock, Paper, Scissors, and Nothing. Each class is trained with images of me with the correct gesture (or no gesture for the Nothing class).

The model was then downloaded as a Tensorflow model as a folder containing the model as  `keras_model.h5`,  and labels `labels.txt`. These will be used in when using our camera for the `user_choice` input when playing "Rock, Paper, Scissors".

## camera_rps.py

>To install the exact dependancies to run the Game in camera_rps, run  `pip install requirements.txt`

camera_rps.py contains the computer vision Rock, Paper, Scissors game. 

### Using the keras model for user_choice 
First, the keras model is loaded in and the camera captures the images of the user. Each image of the live video is normalised and processed such that the keras model can be applied to predict the class of the image being presented to it.

The variable `prediction` contains the output of the model, and each element in the output corresponds to the probability of the input image representing a particular class.

For example, if the prediction has the following output: [[0.8, 0.1, 0.05, 0.05]], there is an 80% chance that the input image shows rock, a 10% chance that it shows paper, a 5% chance that it shows scissors, and a 5% chance that it shows nothing.

The `user_choice` is the value in the prediction with the highest probability.

### Displaying Text using cv2 
Adding text at the correct time to the window (such as having instructions such as pressing [q] to quit) is important for a good user experience.

Texts that were displayed in the window include:
- Live prediction and probability (eg Rock with confidence 0.98)
- Instructions e.g. "First to 3 points wins" and what keys to press for the game
- Current scores
- A countdown, and the game being played that round (e.g. Paper vs Scissors)
- Results of the round and the overall game: These are colourful, especially if the user wins!


The scores are updated each round. Once either the computer or user reaches 3 points, the game ends displaying the result of that game in rainbow text, and resets the scores.

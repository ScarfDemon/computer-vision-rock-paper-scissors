# Computer Vision RPS
In this project a trained computer vision model was produced using [Teachable-Machine](https://teachablemachine.withgoogle.com/). Using this model, we can play a game of 'Rock, Paper, Scissors' by showing a hand gesture to the camera.

## Training image project model using Teachable-Machine
Using [Teachable-Machine](https://teachablemachine.withgoogle.com/), four classes are made: Rock, Paper, Scissors, and Nothing. Each class is trained with images of me with the correct gesture (or no gesture for the Nothing class).

The model was then downloaded as a Tensorflow model as a folder containing the model as  `keras_model.h5`,  and labels `labels.txt`.



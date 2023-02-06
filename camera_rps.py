# %%
import cv2
from keras.models import load_model
import numpy as np
import time
import random

model = load_model('keras_model.h5')

cap = cv2.VideoCapture(0)

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# %%

def display_window(cap, data): # Opens window and camera
    ret, frame = cap.read()
    resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    image_np = np.array(resized_frame)
    normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
    data[0] = normalized_image
    return frame

def get_prediction(): # predicts what action the user has chosen currently
    global model, data
    prediction = model.predict(data).tolist()
    probabilities = dict(zip(["Rock", "Scissors", "Paper", "Nothing"], *prediction))
    probability = round(max(probabilities.values()), 2)
    predicted_user_choice = max(probabilities, key = probabilities.get)
    return predicted_user_choice, probability

def display_info(round = False): # displays current scores in the window
    global computer_score, user_score, font
    cv2.putText(frame, f"You're choice: {predicted_user_choice}  Confidence: {probability}", (50,50), font ,  1, (192, 192, 192), 2, cv2.LINE_AA)
    cv2.putText(frame, "First to 3 points wins!", (925,30), font ,  1, (192, 192, 192), 1, cv2.LINE_AA)
    if round == False:
        cv2.putText(frame, "  Press [S] to start", (950,70), font ,  1, (192, 192, 192), 1, cv2.LINE_AA)
        cv2.putText(frame, "      or [Q] to quit", (950,110), font ,  1, (192, 192, 192), 1, cv2.LINE_AA)
        cv2.putText(frame, "You  | Computer", (1000, 600), font ,  1, (192, 192, 192), 2, cv2.LINE_AA)
        cv2.putText(frame, "________________", (990, 610), font ,  1, (192, 192, 192), 2, cv2.LINE_AA)
        cv2.putText(frame, f"  {user_score}   |   {computer_score}", (987, 645), font ,  1, (192, 192, 192), 2, cv2.LINE_AA)
    

def get_user_choice(): # used to get the user's choice after the countdown for a round
    global predicted_user_choice, probability
    final_user_choice = predicted_user_choice
    final_user_choice_confidence = probability
    return final_user_choice, final_user_choice_confidence

def get_computer_choice(): # randomly chooses computer's choice for a round
    return random.choice(["Rock", "Paper", "Scissors"])

def display_text_countdown(text, t): # Used to display coundown text saying Rock, Paper, Scissors, Shoot! at the correct time
    global t_init, frame, options
    if (t < (time.time() - t_init) < (t+1)) and  ((time.time() - t_init) < (time_limit - 2)):
        cv2.putText(frame, text, (100,400), cv2.FONT_HERSHEY_SIMPLEX ,  2, (192, 192, 192), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.waitKey(2)

def get_winner(final_user_choice, computer_choice): # returns winner and displays text for winner of that round
    global frame, font
    P1, P2 = final_user_choice, computer_choice
    if P1 == P2:
        cv2.putText(frame, "It's a tie!", (50, 400), font ,  3, (192, 192, 192), 2, cv2.LINE_AA)
        winner = "None"
    elif P1 == "Nothing":
        cv2.putText(frame, "No choice detected, please try again!", (50, 400), font ,  1, (192, 192, 192), 1, cv2.LINE_AA)
        winner = "None"
    elif (P1=="Rock" and P2=="Scissors") or (P1=="Paper" and P2=="Rock") or (P1=="Scissors" and P2=="Paper"):
        cv2.putText(frame, "You won!", (50, 400), font ,  3, (192, 192, 192), 2, cv2.LINE_AA)
        winner = "user"
    else:
        cv2.putText(frame, "You lost :(", (50, 400), font ,  3, (192, 192, 192), 2, cv2.LINE_AA)
        winner = "computer"
    return winner

def winner_text(): # displays text of overall winner of game
    global frame, font, user_score, computer_score
    if (user_score == 3):
        cv2.putText(frame, "Congratulations!", (50, 300), font ,  3, (192, 192, 192), 2, cv2.LINE_AA)
        cv2.putText(frame, "  You won the game!", (50, 400), font ,  3, (192, 192, 192), 2, cv2.LINE_AA)
    elif (computer_score == 3):
        cv2.putText(frame, "The computer won ", (50, 300), font ,  3, (192, 192, 192), 2, cv2.LINE_AA)
        cv2.putText(frame, "  the game this time!", (50, 400), font ,  3, (192, 192, 192), 2, cv2.LINE_AA)

def reset_scores(starting_score = 2): # resets score eg at the end of a game
    global computer_score, user_score
    computer_score = starting_score
    user_score = starting_score

# %%

reset_scores()

while True: 

    keys = cv2.waitKey(1) & 0xFF
    frame = display_window(cap, data)

    font = cv2.FONT_HERSHEY_SIMPLEX
    predicted_user_choice, probability = get_prediction()

    display_info()
    
    if keys == ord('s'): # press s to start round
        
        t_init = time.time() # the time at the start of the round
        time_limit = 9 # the round is 9 seconds long
        computer_choice = get_computer_choice()

        while True:

            frame = display_window(cap, data)
            predicted_user_choice, probability = get_prediction()
            display_info(round = True)

            # at 9 seconds, end the round, break out of while loop
            if (time.time() - t_init) > time_limit: 
                break

            # between 7-9 seconds, display text for the winner of the round
            elif (time.time() - t_init) > (time_limit - 2): 
                winner = get_winner(final_user_choice, computer_choice)
                cv2.imshow('frame', frame)
                cv2.waitKey(2)
            
            # between 6-7 seconds, display user's choice vs computer's choice
            elif (time.time() - t_init) > (time_limit - 3): 
                cv2.putText(frame, f"{final_user_choice} vs {computer_choice}", (50, 300), font ,  3, (192, 192, 192), 2, cv2.LINE_AA)
                cv2.imshow('frame', frame)
                cv2.waitKey(2)   
            
            # at 5 seconds (on "Shoot"), record the users choice as their choice for this round
            elif (time.time() - t_init) > (time_limit - 4): 
                final_user_choice, final_user_choice_confidence = get_user_choice()
            
            # from 0 to 5 seconds after the round was started, display Rock Paper Scissors Shoot countdown
            elif (time.time() - t_init) >= 0: 
                options = ["On shoot, ready?", "Rock", "Paper", "Scissors", "Shoot!"]
                for i in range(len(options)):
                    display_text_countdown(options[i], i)

        # add 1 to the score of the winner
        if winner == "user": 
            user_score += 1
        elif winner == "computer":
            computer_score += 1
        display_info()
    cv2.imshow('frame', frame)

    # end game at 3 points and display text of the winner of the entire game
    if (user_score == 3) or (computer_score == 3): 
        t_init = time.time()
        
        # put overall winner text on for 2 seconds
        while (time.time()-t_init) <= 2: 
            winner_text()
            cv2.imshow('frame', frame)
            cv2.waitKey(2)
            if (time.time()-t_init) > 2:
                break
        reset_scores()

    # Press q to close the window
    if keys == ord('q'):
        break
            
# After the loop release the cap object
cap.release()

# Destroy all the windows
cv2.destroyAllWindows()
    
# %%
    



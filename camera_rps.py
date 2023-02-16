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
class RPS():
    
    def __init__(self):
        self.user_wins = 0
        self.computer_wins = 0
        # for rainbow text
        self.colours = [(255,0,0), (255,125,0), (255,255,0), (125,255,0), (0,255,0), (0,255,125), (0,255,255), (0,125,255), (0,0,225), (125,0,255), (255,0,255), (225,0,125)] 

    def get_prediction(self): # predicts what action the user has chosen currently
        prediction = model.predict(data).tolist()
        probabilities = dict(zip(["Rock", "Paper", "Scissors", "Nothing"], *prediction))
        probability = round(max(probabilities.values()), 2)
        predicted_user_choice = max(probabilities, key = probabilities.get)
        return predicted_user_choice, probability

    def get_computer_choice(self): # randomly chooses computer's choice for a round
        return random.choice(["Rock", "Paper", "Scissors"])

    def display_score(self):
        cv2.putText(self.frame, "You  | Computer", (1000, 600), cv2.FONT_HERSHEY_SIMPLEX ,  1, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(self.frame, "________________", (990, 610), cv2.FONT_HERSHEY_SIMPLEX ,  1, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(self.frame, f"  {self.user_wins}   |   {self.computer_wins}", (987, 645), cv2.FONT_HERSHEY_SIMPLEX ,  1, (0,0,0), 2, cv2.LINE_AA)

    def display_info(self, round = False): # displays current scores in the window
        predicted_user_choice, probability = self.get_prediction()
        cv2.putText(self.frame, f"You're choice: {predicted_user_choice}  Confidence: {probability}", (50,50), cv2.FONT_HERSHEY_SIMPLEX ,  1, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(self.frame, "First to 3 points wins!", (925,30), cv2.FONT_HERSHEY_SIMPLEX ,  1, (0,0,0), 1, cv2.LINE_AA)
        
        if round == False:
            self.display_score()
            cv2.putText(self.frame, "  Press [S] to start", (950,70), cv2.FONT_HERSHEY_SIMPLEX ,  1, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(self.frame, "      or [Q] to quit", (950,110), cv2.FONT_HERSHEY_SIMPLEX ,  1, (0,0,0), 1, cv2.LINE_AA)
        cv2.imshow('frame', self.frame)

    def display_window(self, cap, data): # Opens window and camera
        ret, self.frame = cap.read()
        resized_frame = cv2.resize(self.frame, (224, 224), interpolation = cv2.INTER_AREA)
        image_np = np.array(resized_frame)
        normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
        data[0] = normalized_image
        return self.frame

    def winner_text(self): # displays text of overall winner of game
        
        for colour in self.colours: # rainbow text
            if self.user_wins == 3:
                cv2.putText(self.frame, "Congratulations!", (50, 300), cv2.FONT_HERSHEY_SIMPLEX ,  3, colour, 2, cv2.LINE_AA)
                cv2.putText(self.frame, "  You won the game!", (50, 400), cv2.FONT_HERSHEY_SIMPLEX ,  3, colour, 2, cv2.LINE_AA)
            elif self.computer_wins == 3:
                cv2.putText(self.frame, "The computer won ", (50, 300), cv2.FONT_HERSHEY_SIMPLEX ,  3, colour, 2, cv2.LINE_AA)
                cv2.putText(self.frame, "  the game this time!", (50, 400), cv2.FONT_HERSHEY_SIMPLEX ,  3, colour, 2, cv2.LINE_AA)
            cv2.imshow('frame', self.frame)
            cv2.waitKey(2)
        
    def get_winner(self, final_user_choice, computer_choice): # returns winner and displays text for winner of that round
        P1, P2 = final_user_choice, computer_choice
        if P1 == P2:
            cv2.putText(self.frame, "It's a tie!", (50, 400), cv2.FONT_HERSHEY_SIMPLEX ,  3, (0,0,0), 2, cv2.LINE_AA)
            winner = "None"
        elif P1 == "Nothing":
            cv2.putText(self.frame, "No choice detected, please try again!", (50, 400), cv2.FONT_HERSHEY_SIMPLEX ,  1, (0,0,0), 1, cv2.LINE_AA)
            winner = "None"
        elif (P1=="Rock" and P2=="Scissors") or (P1=="Paper" and P2=="Rock") or (P1=="Scissors" and P2=="Paper"):
            for colour in self.colours: # for rainbow text
                cv2.putText(self.frame, "You won!", (50, 400), cv2.FONT_HERSHEY_SIMPLEX ,  3, colour, 2, cv2.LINE_AA)
                cv2.imshow('frame', self.frame)
                cv2.waitKey(2)
            winner = "user"
            
        else:
            cv2.putText(self.frame, "You lost :(", (50, 400), cv2.FONT_HERSHEY_SIMPLEX ,  3, (0,0,0), 2, cv2.LINE_AA)
            winner = "computer"
        
        return winner

    def reset_scores(self, starting_scores = 2): # resets score eg at the end of a game
        self.computer_wins = starting_scores
        self.user_wins = starting_scores
        return self.computer_wins, self.user_wins

    def RPS_round(self): # runs a round of RPS: counts down, states game, winner and updates scores
        self.t_init = time.time() # the time at the start of the round
        self.time_per_round = 9 # the round is 9 seconds long
        self.computer_choice = self.get_computer_choice()

        while True:

            self.frame = self.display_window(cap, data)
            predicted_user_choice, probability = self.get_prediction()
            self.display_info(round = True)

            # at 9 seconds, end the round, break out of while loop
            if (time.time() - self.t_init) > self.time_per_round:
                break

            # between 7-9 seconds, display text for the winner of the round
            elif (time.time() - self.t_init) > (self.time_per_round - 2):
                winner = self.get_winner(final_user_choice, self.computer_choice)
                cv2.imshow('frame', self.frame)
                cv2.waitKey(2)
            
            # between 6-7 seconds, display user's choice vs computer's choice
            elif (time.time() - self.t_init) > (self.time_per_round - 3):
                print(time.time() - self.t_init)
                cv2.putText(self.frame, f"{final_user_choice} vs {self.computer_choice}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX ,  3, (0,0,0), 2, cv2.LINE_AA)
                cv2.imshow('frame', self.frame)
                cv2.waitKey(2)   

            # at 5 seconds (on "Shoot"), record the users choice as their choice for this round
            elif (time.time() - self.t_init) > (self.time_per_round - 4):
                final_user_choice, final_user_choice_confidence = predicted_user_choice, probability # get_user_choice
            
            # from 0 to 5 seconds after the round was started, display Rock Paper Scissors Shoot countdown
            elif (time.time() - self.t_init) >= 0:
                options = ["On shoot, ready?", "Rock", "Paper", "Scissors", "Shoot!"]

                def display_text_countdown(text, t): # Used to display coundown text saying Rock, Paper, Scissors, Shoot! at the correct time
                    if (t < (time.time() - self.t_init) < (t+1)) and  ((time.time() - self.t_init) < (self.time_per_round - 2)):
                        cv2.putText(self.frame, text, (100,400), cv2.FONT_HERSHEY_SIMPLEX ,  2, (0,0,0), 2, cv2.LINE_AA)
                        cv2.imshow('frame', self.frame)
                        cv2.waitKey(2)
                
                for i in range(len(options)):
                    display_text_countdown(options[i], i)

        # add 1 to the score of the winner
        if winner == "user":
            self.user_wins += 1
        elif winner == "computer":
            self.computer_wins += 1
        self.display_score()
    
    def end_game(self): 
        t_init = time.time()

        # put overall winner text on for 2 seconds
        while (time.time()-t_init) <= 2: 
            self.winner_text()
            if (time.time()-t_init) > 2:
                break
        self.reset_scores(0) # reset the scores at the end of the game
    
    
# %%

def play_game():
    play = RPS()
    play.reset_scores()

    while True: 
        keys = cv2.waitKey(1) & 0xFF
        frame = play.display_window (cap, data)
        play.display_info() # display game info on window
        
        if keys == ord('s'): # press s to start round
            play.RPS_round() # play round

        cv2.imshow('frame', frame)
        # end game at 3 points and display text of the winner of the entire game
        if (play.user_wins == 3) or (play.computer_wins == 3): 
            play.end_game()
        
        # Press q to close the window
        if keys == ord('q'):
            break
                
    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    play_game()


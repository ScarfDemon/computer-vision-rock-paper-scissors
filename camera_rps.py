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
        self.time_per_round = 9
        self.user_score = 0
        self.computer_score = 0
        #self.options = ["Rock", "Paper", "Scissors", "Nothing"]
        

    def get_prediction(self):
        prediction = model.predict(data).tolist()
        probabilities = dict(zip(["Rock", "Scissors", "Paper", "Nothing"], *prediction))
        probability = round(max(probabilities.values()), 2)
        predicted_user_choice = max(probabilities, key = probabilities.get)
        return predicted_user_choice, probability

    def get_user_choice(self, predicted_user_choice, probability):
        final_user_choice = predicted_user_choice
        final_user_choice_confidence = probability
        return final_user_choice, final_user_choice_confidence

    def get_computer_choice(self):
        return random.choice(["Rock", "Paper", "Scissors"])

    def display_score(self):
        cv2.putText(self.frame, "You  | Computer", (1000, 600), cv2.FONT_HERSHEY_SIMPLEX ,  1, (192, 192, 192), 2, cv2.LINE_AA)
        cv2.putText(self.frame, "________________", (990, 610), cv2.FONT_HERSHEY_SIMPLEX ,  1, (192, 192, 192), 2, cv2.LINE_AA)
        cv2.putText(self.frame, f"  {self.user_score}   |   {self.computer_score}", (987, 645), cv2.FONT_HERSHEY_SIMPLEX ,  1, (192, 192, 192), 2, cv2.LINE_AA)

    def display_info(self, round = False): # displays current scores in the window
        predicted_user_choice, probability = self.get_prediction()
        cv2.putText(self.frame, f"You're choice: {predicted_user_choice}  Confidence: {probability}", (50,50), cv2.FONT_HERSHEY_SIMPLEX ,  1, (192, 192, 192), 2, cv2.LINE_AA)
        cv2.putText(self.frame, "First to 3 points wins!", (925,30), cv2.FONT_HERSHEY_SIMPLEX ,  1, (192, 192, 192), 1, cv2.LINE_AA)
        if round.all() == False:
            cv2.putText(self.frame, "  Press [S] to start", (950,70), cv2.FONT_HERSHEY_SIMPLEX ,  1, (192, 192, 192), 1, cv2.LINE_AA)
            cv2.putText(self.frame, "      or [Q] to quit", (950,110), cv2.FONT_HERSHEY_SIMPLEX ,  1, (192, 192, 192), 1, cv2.LINE_AA)
            cv2.putText(self.frame, "You  | Computer", (1000, 600), cv2.FONT_HERSHEY_SIMPLEX ,  1, (192, 192, 192), 2, cv2.LINE_AA)
            cv2.putText(self.frame, "________________", (990, 610), cv2.FONT_HERSHEY_SIMPLEX ,  1, (192, 192, 192), 2, cv2.LINE_AA)
            cv2.putText(self.frame, f"  {self.user_score}   |   {self.computer_score}", (987, 645), cv2.FONT_HERSHEY_SIMPLEX ,  1, (192, 192, 192), 2, cv2.LINE_AA)
        cv2.imshow('frame', self.frame)

    def display_window(self, cap, data):
        ret, self.frame = cap.read()
        resized_frame = cv2.resize(self.frame, (224, 224), interpolation = cv2.INTER_AREA)
        image_np = np.array(resized_frame)
        normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
        data[0] = normalized_image

        return self.frame

    def winner_text(self):
        #global frame
        if self.user_score == 3:
            cv2.putText(self.frame, "Congratulations!", (50, 300), cv2.FONT_HERSHEY_SIMPLEX ,  3, (192, 192, 192), 2, cv2.LINE_AA)
            cv2.putText(self.frame, "  You won the game!", (50, 400), cv2.FONT_HERSHEY_SIMPLEX ,  3, (192, 192, 192), 2, cv2.LINE_AA)
        elif self.computer_score == 3:
            cv2.putText(self.frame, "The computer won ", (50, 300), cv2.FONT_HERSHEY_SIMPLEX ,  3, (192, 192, 192), 2, cv2.LINE_AA)
            cv2.putText(self.frame, "  the game this time!", (50, 400), cv2.FONT_HERSHEY_SIMPLEX ,  3, (192, 192, 192), 2, cv2.LINE_AA)

    def get_winner(self, final_user_choice, computer_choice):
        #global frame
        P1, P2 = final_user_choice, computer_choice
        if P1 == P2:
            cv2.putText(self.frame, "It's a tie!", (50, 400), cv2.FONT_HERSHEY_SIMPLEX ,  3, (192, 192, 192), 2, cv2.LINE_AA)
            winner = "None"
        elif P1 == "Nothing":
            cv2.putText(self.frame, "No choice detected, please try again!", (50, 400), cv2.FONT_HERSHEY_SIMPLEX ,  1, (192, 192, 192), 1, cv2.LINE_AA)
            winner = "None"
        elif (P1=="Rock" and P2=="Scissors") or (P1=="Paper" and P2=="Rock") or (P1=="Scissors" and P2=="Paper"):
            cv2.putText(self.frame, "You won!", (50, 400), cv2.FONT_HERSHEY_SIMPLEX ,  3, (192, 192, 192), 2, cv2.LINE_AA)
            winner = "user"
        else:
            cv2.putText(self.frame, "You lost :(", (50, 400), cv2.FONT_HERSHEY_SIMPLEX ,  3, (192, 192, 192), 2, cv2.LINE_AA)
            winner = "computer"
        return winner

    def reset_scores(self, starting_scores = 2):
        self.computer_score = starting_scores
        self.user_score = starting_scores
        return self.computer_score, self.user_score

    # def RPS_round(self):
    #     self.t_init = time.time()
    #     self.time_per_round = 9
    #     self.computer_choice = self.get_computer_choice()

    #     def display_text_countdown(text, t):
    #         global t_init, frame
    #         if (t < (time.time() - t_init) < (t+1)) and  ((time.time() - t_init) < (self.time_per_round - 2)):
    #             cv2.putText(frame, text, (100,400), cv2.FONT_HERSHEY_SIMPLEX ,  2, (192, 192, 192), 2, cv2.LINE_AA)
    #             cv2.imshow('frame', frame)
    #             cv2.waitKey(2)

    #     while True:

    #         frame = self.display_window(cap, data)
    #         predicted_user_choice, probability = self.get_prediction()
    #         self.display_info(round = True)

    #         if (time.time() - t_init) > self.time_per_round:
    #             break
    #         elif (time.time() - t_init) > (self.time_per_round - 2):
    #             winner = self.get_winner(final_user_choice, computer_choice)
    #             cv2.imshow('frame', frame)
    #             cv2.waitKey(2)
    #         elif (time.time() - t_init) > (self.time_per_round - 3):
    #             print(time.time() - t_init)
    #             cv2.putText(frame, f"{final_user_choice} vs {computer_choice}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX ,  3, (192, 192, 192), 2, cv2.LINE_AA)
    #             cv2.imshow('frame', frame)
    #             cv2.waitKey(2)   
    #         elif (time.time() - t_init) > (self.time_per_round - 4):
    #             final_user_choice, final_user_choice_confidence = self.get_user_choice(predicted_user_choice, probability)
    #         elif (time.time() - t_init) >= 0:
    #             options = ["On shoot, ready?", "Rock", "Paper", "Scissors", "Shoot!"]
    #             for i in range(len(options)):
    #                 display_text_countdown(options[i], i)

    #     if winner == "user":
    #         self.user_score += 1
    #     elif winner == "computer":
    #         self.computer_score += 1
    #     self.display_score(self.computer_score, self.user_score)
    
    # def end_game(self):
    #     t_init = time.time()
    #     broken = False
    #     while ((time.time()-t_init) <= 2) and (broken == False):
    #         self.winner_text()
    #         cv2.imshow('frame', frame)
    #         cv2.waitKey(2)
    #         if (time.time()-t_init) > 2:
    #             broken = True
    #         if broken == True:
    #             self.computer_score, self.user_score = self.reset_scores()
    #             break
    



    
# %%

def play_game():
    play = RPS()
    #play.reset_scores()

    while True: 
        keys = cv2.waitKey(1) & 0xFF

        frame = play.display_window (cap, data)

        play.display_info(frame)

        play.display_score()
        
        # if keys == ord('s'):
        #     play.RPS_round()

        # cv2.imshow('frame', frame)
        
        # if (user_score == 3) or (computer_score == 3):
        #     play.end_game()
        # # Press q to close the window
        if keys == ord('q'):
            break
                
    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    play_game()


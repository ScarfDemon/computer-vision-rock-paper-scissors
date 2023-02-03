

# %%
import cv2
from keras.models import load_model
import numpy as np
import time

model = load_model('keras_model.h5')

cap = cv2.VideoCapture(0)

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# %%
def get_prediction():
    prediction = model.predict(data)
    prediction = prediction.tolist()
    probabilities = dict(zip(["Rock", "Scissors", "Paper", "Nothing"], *prediction))
    probability = round(max(probabilities.values()), 2)
    predicted_user_choice = max(probabilities, key = probabilities.get)
    return predicted_user_choice, probability

def get_user_choice():
    global predicted_user_choice, probability
    final_user_choice = predicted_user_choice
    final_user_choice_confidence = probability
    return final_user_choice, final_user_choice_confidence

def display_score():
    global computer_score, user_score
    cv2.putText(frame, "You  | Computer", (1000, 600), font ,  1, (192, 192, 192), 2, cv2.LINE_AA)
    cv2.putText(frame, "________________", (990, 610), font ,  1, (192, 192, 192), 2, cv2.LINE_AA)
    cv2.putText(frame, f"  {user_score}   |   {computer_score}", (987, 645), font ,  1, (192, 192, 192), 2, cv2.LINE_AA)

def display_text_time(text, t):
    global t_init, frame
    if (t < (time.time() - t_init) < (t+1)) and  ((time.time() - t_init) < (time_limit - 2)):
        cv2.putText(frame, text, (50,100), cv2.FONT_HERSHEY_SIMPLEX ,  1, (192, 192, 192), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.waitKey(2)
    else:
        pass

def display_window(cap, data):
            ret, frame = cap.read()
            resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
            image_np = np.array(resized_frame)
            normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
            data[0] = normalized_image
            return frame
# %%

computer_score = 2
user_score = 2

while True: 
    keys = cv2.waitKey(1) & 0xFF
    frame = display_window(cap, data)

    font = cv2.FONT_HERSHEY_SIMPLEX
    predicted_user_choice, probability = get_prediction()
    cv2.putText(frame, f"You're choice: {predicted_user_choice}  Confidence: {probability}", (50,50), font ,  1, (192, 192, 192), 2, cv2.LINE_AA)
    cv2.putText(frame, "First to 3 points wins!", (925,30), font ,  1, (192, 192, 192), 1, cv2.LINE_AA)
    cv2.putText(frame, "  Press [S] to start", (950,70), font ,  1, (192, 192, 192), 1, cv2.LINE_AA)
    cv2.putText(frame, "      or [Q] to quit", (950,110), font ,  1, (192, 192, 192), 1, cv2.LINE_AA)
    
    print(predicted_user_choice)

    
    display_score()
    
    if keys == ord('s'):
        import manual_rps
        t_init = time.time()
        time_limit = 8
        t = 0
        computer_choice = manual_rps.get_computer_choice()
        broken = False
        #display_score(computer_score, user_score)
        while broken==False:
            
            frame = display_window(cap, data)
            predicted_user_choice, probability = get_prediction()
            display_score()
            cv2.putText(frame, f"You're choice: {predicted_user_choice}  Confidence: {probability}", (50,50), font ,  1, (192, 192, 192), 2, cv2.LINE_AA)
            cv2.putText(frame, "First to 3 points wins!", (925,30), font ,  1, (192, 192, 192), 1, cv2.LINE_AA)
            #display_score(computer_score, user_score)
            if (time.time() - t_init) > time_limit:
                broken = True
            elif (time.time() - t_init) > (time_limit - 2):
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
                cv2.imshow('frame', frame)
                cv2.waitKey(2)
            elif (time.time() - t_init) > (time_limit - 3):
                print(time.time() - t_init)
                cv2.putText(frame, f"{final_user_choice} vs {computer_choice}", (50, 300), font ,  3, (192, 192, 192), 2, cv2.LINE_AA)
                cv2.imshow('frame', frame)
                cv2.waitKey(2)   
            elif (time.time() - t_init) > (time_limit - 4):
                final_user_choice, final_user_choice_confidence = get_user_choice()
            elif (time.time() - t_init) >= 0:
                options = ["Rock", "Paper", "Scissors", "Shoot!", "Shoot!"]
                for i in range(len(options)):
                    display_text_time(options[i], i)

                        
            else:
                continue
            if broken == True:
                break
        if winner == "user":
            user_score += 1
        elif winner == "computer":
            computer_score += 1
        else:
            None
        display_score()
        #continue
            

    cv2.imshow('frame', frame)


    
    if (user_score == 3) or (computer_score == 3):
        t_init = time.time()
        broken = False
        while ((time.time()-t_init) <= 2) and (broken == False):
            if (user_score == 3):
                cv2.putText(frame, "Congratulations!", (50, 300), font ,  3, (192, 192, 192), 2, cv2.LINE_AA)
                cv2.putText(frame, " You won the game!", (50, 400), font ,  3, (192, 192, 192), 2, cv2.LINE_AA)
                
            elif (computer_score == 3):
                cv2.putText(frame, "The computer won ", (50, 300), font ,  3, (192, 192, 192), 2, cv2.LINE_AA)
                cv2.putText(frame, "the game this time!", (50, 400), font ,  3, (192, 192, 192), 2, cv2.LINE_AA)
            else:
                continue
            
            cv2.imshow('frame', frame)
            cv2.waitKey(2)

            if (time.time()-t_init) > 2:
                broken = True
            
            if broken == True:
                computer_score = 2
                user_score = 2
                break


    # Press q to close the window
    if keys == ord('q'):
        break
            
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
    
# %%
    



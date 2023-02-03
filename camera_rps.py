

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
    global predicted_user_choice
    global probability
    final_user_choice = predicted_user_choice
    final_user_choice_confidence = probability
    cv2.putText(frame, f"{final_user_choice}, Confidence: {final_user_choice_confidence}", (400, 100), font ,  1, (192, 192, 192), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    cv2.waitKey(100) 
    return final_user_choice, final_user_choice_confidence

# %%
def display_text_time(text, t):
    global t_init
    global frame
    if (t < (time.time() - t_init) < (t+1)) and  ((time.time() - t_init) < (time_limit - 2)):
        cv2.putText(frame, text, (100,100), cv2.FONT_HERSHEY_SIMPLEX ,  1, (192, 192, 192), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.waitKey(2)
    else:
        pass

while True: 
    keys = cv2.waitKey(1) & 0xFF
    ret, frame = cap.read()
    resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    image_np = np.array(resized_frame)
    normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
    data[0] = normalized_image
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    predicted_user_choice, probability = get_prediction()
    cv2.putText(frame, f"You're choice: {predicted_user_choice} with a confidence of {probability}", (50,50), font ,  1, (192, 192, 192), 2, cv2.LINE_AA)

    print(predicted_user_choice)
    if keys == ord('a'):
        import manual_rps
        t_init = time.time()
        time_limit = 8
        t = 0
        computer_choice = manual_rps.get_computer_choice()
        broken = False
        
        while broken==False:
            ret, frame = cap.read()
            resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
            image_np = np.array(resized_frame)
            normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
            data[0] = normalized_image
            predicted_user_choice, probability = get_prediction()
            cv2.putText(frame, f"You're choice: {predicted_user_choice} with a confidence of {probability}", (50,50), font ,  1, (192, 192, 192), 2, cv2.LINE_AA)

            if (time.time() - t_init) > time_limit:
                broken = True
            elif (time.time() - t_init) > (time_limit - 2):
                P1, P2 = final_user_choice, computer_choice
                if P1 == P2:
                    cv2.putText(frame, "It's a tie!", (200, 500), font ,  3, (192, 192, 192), 2, cv2.LINE_AA)
                elif (P1=="Rock" and P2=="Scissors") or (P1=="Paper" and P2=="Rock") or (P1=="Scissors" and P2=="Paper"):
                    cv2.putText(frame, "You won!", (200, 500), font ,  3, (192, 192, 192), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "You lost :(", (200, 500), font ,  3, (192, 192, 192), 2, cv2.LINE_AA)
                cv2.imshow('frame', frame)
                cv2.waitKey(100)
            elif (time.time() - t_init) > (time_limit - 3):
                print(time.time() - t_init)
                cv2.putText(frame, f"{final_user_choice} vs {computer_choice}", (200, 500), font ,  3, (192, 192, 192), 2, cv2.LINE_AA)
                cv2.imshow('frame', frame)
                cv2.waitKey(100)   
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
        #continue
            

    cv2.imshow('frame', frame)
    # Press q to close the window
    if keys == ord('q'):
        break
            
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
    
# %%
    


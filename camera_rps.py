

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
    probability = max(probabilities.values())
    predicted_user_choice = max(probabilities, key = probabilities.get)
    return predicted_user_choice, probability



# %%


while True: 
    keys = cv2.waitKey(1) & 0xFF
    ret, frame = cap.read()
    resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    image_np = np.array(resized_frame)
    normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
    data[0] = normalized_image
    
    predicted_user_choice, probability = get_prediction()
    cv2.putText(frame, f"You're choice: {predicted_user_choice} with a confidence of {probability}", (50,50), cv2.FONT_HERSHEY_SIMPLEX ,  1, (192, 192, 192), 2, cv2.LINE_AA)

    print(predicted_user_choice)
    if keys == ord('a'):
        import manual_rps
        time_limit = 0
        t = time_limit
        t_init = time.time()
        computer_choice = manual_rps.get_computer_choice()
        #cv2.putText(frame, "HELLO THERE", (600,200), cv2.FONT_HERSHEY_SIMPLEX ,  4, (192, 192, 192), 4, cv2.LINE_AA)
        broken = False
        while broken==False:
            if (time.time() - t_init) > 5:
                broken = True
            elif (time.time() - t_init) > 3:
                print(time.time()-t_init)
                final_predicted_choice = predicted_user_choice
                predicted_user_choice_confidence = probability
                print(final_predicted_choice, "    ", predicted_user_choice_confidence)
                cv2.putText(frame, f"{final_predicted_choice} vs {computer_choice}", (300, 500), font ,  3, (192, 192, 192), 2, cv2.LINE_AA)
                #cv2.imshow('frame', frame)
                cv2.waitKey(500)
                #broken = True
            elif (time.time() - t_init) >= 0:
                print(time.time()-t_init)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, "Rock, Paper, Scissors, Shoot!", (200,200), font ,  2, (192, 192, 192), 2, cv2.LINE_AA)
                #cv2.imshow('frame', frame)
                print(t)
                t += 1
                cv2.waitKey(500)
                continue
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
    


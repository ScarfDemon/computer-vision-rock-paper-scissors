

# %%
import cv2
from keras.models import load_model
import numpy as np
model = load_model('keras_model.h5')

cap = cv2.VideoCapture(0)

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# %%
def get_prediction():
    global prediction
    #print(prediction)
    probabilities = dict(zip(["Rock", "Scissors", "Paper", "Nothing"], *prediction))
    #print("probabilities  ",probabilities)
    probability = max(probabilities.values())
    #print("probability  ", probability)
    computer_choice = max(probabilities, key = probabilities.get)
    #print("comp choice  ", computer_choice)
    return computer_choice, probability


# %%
while True: 
    ret, frame = cap.read()
    resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    image_np = np.array(resized_frame)
    normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
    data[0] = normalized_image
    prediction = model.predict(data)
    prediction = prediction.tolist()
    #print(type(prediction))
    cv2.imshow('frame', frame)
    # Press q to close the window
    computer_choice, probability = get_prediction()
    print(computer_choice, "    ", probability)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
    
# %%


import cv2
import matplotlib.pyplot as plt
import os
from deepface import DeepFace

# Get the directory where main.py is located
# script_dir = os.path.dirname(os.path.abspath(__file__))
# img_path = os.path.join(script_dir, "assets", "happy.jpg")



img = cv2.imread("Back-End/assets/happy.jpg")



predictions = DeepFace.analyze(img)
# print(predictions) #this is use to print the entire stats

print(predictions[0]["dominant_emotion"])

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#print (faceCascade.empty())
faces = faceCascade.detectMultiScale(gray, 1.1,4)
# Draw a rectangle around the faces
for(x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)



font = cv2.FONT_HERSHEY_SIMPLEX
# Use putText() method for
# inserting text on video
cv2.putText(img,predictions[0]['dominant_emotion'],
(50, 50),
font, 3,
(0, 0, 255),
3,
cv2.LINE_4)




img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
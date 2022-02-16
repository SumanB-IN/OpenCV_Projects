import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
image_colr = cv2.imread("image_multi.jpg", 1)
image_gray = cv2.cvtColor(image_colr, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(image_colr, scaleFactor=1.2, minNeighbors=10)

print(type(faces))
print(faces)

for x, y, w, h in faces:
    image = cv2.rectangle(image_colr, (x, y), (x + w, y + h), (255, 0, 0), 3)


# print(image)
# print(image.shape)

cv2.imshow("Suman_Face_detected", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
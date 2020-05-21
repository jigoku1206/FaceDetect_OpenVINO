import cv2 as cv
import time

print(cv.__version__)

# Load the model.
net = cv.dnn_DetectionModel('face-detection-adas-0001.xml',
                            'face-detection-adas-0001.bin')

# Specify target device.
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

cam = cv.VideoCapture(4)
print('OpenCam Complete')

with open('/dev/fb1', 'rb+') as buf:
   while True:
      start = time.clock()
      ret, frame = cam.read()

      # Perform an inference.
      _, confidences, boxes = net.detect(frame, confThreshold=0.5)

      # Draw detected faces on the frame.
      for confidence, box in zip(list(confidences), boxes):
         cv.rectangle(frame, box, color=(0, 255, 0))

      frame32 = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
      fbframe = cv.resize(frame32, (1280, 720))
	  end = time.clock()
	  print(end - start)
      # buf.write(fbframe)
      # buf.seek(0, 0)


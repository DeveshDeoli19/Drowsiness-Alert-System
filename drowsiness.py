from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import playsound
import imutils
import time
import dlib
import cv2
from threading import Thread

def sound_alarm(path):
	playsound.playsound(path)

def eye_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
 
def mouth_ratio(mouth):
	
	A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
	B = dist.euclidean(mouth[4], mouth[8]) # 53, 57
	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

	# compute the mouth aspect ratio
	mar = (A + B) / (2.0 * C)

	# return the mouth aspect ratio
	return mar


MOU_THRESH = 0.75
EYE_THRESH = 0.25
EYE_MOUTH_CONSEC_FRAMES = 40

COUNT = 0
ALARM_ON = False



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)
(mStart, mEnd) = (49, 68)

vs = VideoStream(src=0).start()
time.sleep(1.0)

while True:
	frame = vs.read()

	frame = imutils.resize(frame, width=450)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)
	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		mouth = shape[mStart:mEnd]
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]

		leftEYR = eye_ratio(leftEye)
		rightEYR = eye_ratio(rightEye)
		mouthMAR = mouth_ratio(mouth)

		mar = mouthMAR

		ear = (leftEYR + rightEYR) / 2.0

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		mouthHull = cv2.convexHull(mouth)

		cv2.drawContours(frame, [leftEyeHull],-1, (255, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull],-1, (255, 255, 0), 1)
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

		if ear < EYE_THRESH or mar > MOU_THRESH:
			COUNT += 1
			if COUNT >= EYE_MOUTH_CONSEC_FRAMES:
				if not ALARM_ON:
					ALARM_ON = True
					t = Thread(target=sound_alarm,args=("alarm.wav",))
					t.deamon = True
					t.start()
				cv2.putText(frame, "BE ALERT!", (10, 30),
				cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)
		else:
			COUNT = 0
			ALARM_ON = False
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
		cv2.putText(frame, "MOU: {:.2f}".format(mar), (300, 50),
			cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
 
	cv2.imshow("Front Face Detection", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
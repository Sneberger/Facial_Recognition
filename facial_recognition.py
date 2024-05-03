#pip install dlib==19.22
#pip install opencv_python
#pip install face_recognition

import face_recognition
import os, sys
import cv2
import numpy as np
import math
import cProfile
from threading import Thread
import time


def split_name(name):
    if name == 'Unknown (Unknown)':
        return 'Unknown', 'Unknown'
    parsed = name.split('.', 1)
    name_only = parsed[0]
    parsed_2 = parsed[1].split(' ')
    percentage_only = parsed_2[1]

    # print(f'Parsed Whole: {parsed}')
    # print(f'Name Only: {name_only}')
    # print(f'Percentage Only: {percentage_only}')
    return name_only, percentage_only


def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'
    

class ThreadedCamera(object):   # FAILED ATTEMPT AT THREADING
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
    frames = 0

    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)    # if not video add ', cv2.CAP_DSHOW'
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
       
        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)
        self.encode_faces()
        
        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        
        print(self.known_face_names)
        
    def update(self):
        while True:
            if self.capture.isOpened():
                ret, frame = self.capture.read()

                if self.process_current_frame:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

                    # find all the faces in the current frame
                    self.face_locations = face_recognition.face_locations(rgb_small_frame)
                    self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                    self.face_names = []
                    for face_encoding in self.face_encodings:
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                        name = 'Unknown'
                        confidence = 'Unknown'

                        face_distances = face_recognition.face_distance(self.known_face_encodings,face_encoding)
                        best_match_index = np.argmin(face_distances)

                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            confidence = '50%'
                            #confidence = face_confidence(face_distances[best_match_index])

                        self.face_names.append(f'{name} ({confidence})')

                #self.process_current_frame = not self.process_current_frame    # looks at every other frame

                # display and annotations (x, y, w, h)
                for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    # NOTE BGR not RGB
                    if name == "Unknown (Unknown)":
                        cv2.rectangle(frame, (left - 30, top - 75), (right + 30, bottom + 35), (0, 0, 255), 2)
                        cv2.rectangle(frame, (left - 30, bottom + 35), (right + 30, bottom + 75), (0, 0, 255), -1)
                        cv2.putText(frame, split_name(name)[0], (left - 30, bottom + 60), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                        cv2.rectangle(frame, (left - 30, bottom + 75), (right + 30, bottom + 110), (0, 0, 255), -1)
                        cv2.putText(frame, split_name(name)[1], (left - 10, bottom + 95), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                    else:
                        cv2.rectangle(frame, (left - 30, top - 75), (right + 30, bottom + 35), (0, 255, 0), 2)
                        #cv2.rectangle(frame, (top, right), (top + bottom, bottom + left), (0, 255, 0), 2)
                        #cv2.ellipse(frame, frame.mean(axis=0), (100, 100), 0, 0, 360, (0, 255, 0), 2)
                        cv2.rectangle(frame, (left - 30, bottom + 35), (right + 30, bottom + 75), (0, 255, 0), -1)
                        cv2.putText(frame, split_name(name)[0], (left - 30, bottom + 60), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                        cv2.rectangle(frame, (left - 30, bottom + 75), (right + 30, bottom + 110), (0, 255, 0), -1)
                        cv2.putText(frame, split_name(name)[1], (left - 10, bottom + 95), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                
                cv2.imshow('Face Recognition', frame)

                if cv2.waitKey(1) == ord('q'):
                    self.capture.release()
                    cv2.destroyAllWindows()
                    exit(1)

                time.sleep(self.FPS)
            
    def show_frame(self):
        cv2.imshow('frame', self.frame)
        cv2.waitKey(self.FPS_MS)


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
    frames = 0

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        
        print(self.known_face_names)


    def run_recognition(self):
        video_capture = cv2.VideoCapture('Supermodels_480p.mp4') # argument is which camera: 0=laptop; 1=webcam
                                            # for video files insert 'filename.avi'
                                            # and remove ', cv2.CAP_DSHOW'
        # video_capture = ThreadedCamera(1)     # FAILED ATTEMPT AT THREADING
        # video_capture = video_capture.capture # FAILED ATTEMPT AT THREADING
        
        if not video_capture.isOpened():
            sys.exit('Video source not found . . .')

        while True:
            ret, frame = video_capture.read()

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

                # find all the faces in the current frame
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = 'Unknown'
                    confidence = 'Unknown'

                    face_distances = face_recognition.face_distance(self.known_face_encodings,face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                    self.face_names.append(f'{name} ({confidence})')

            self.process_current_frame = not self.process_current_frame    # looks at every other frame

            # display and annotations (x, y, w, h)
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # NOTE BGR not RGB
                if name == "Unknown (Unknown)":
                    cv2.rectangle(frame, (left - 30, top - 75), (right + 30, bottom + 35), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left - 30, bottom + 35), (right + 30, bottom + 75), (0, 0, 255), -1)
                    cv2.putText(frame, split_name(name)[0], (left - 30, bottom + 60), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                    cv2.rectangle(frame, (left - 30, bottom + 75), (right + 30, bottom + 110), (0, 0, 255), -1)
                    cv2.putText(frame, split_name(name)[1], (left - 10, bottom + 95), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                else:
                    cv2.rectangle(frame, (left - 30, top - 75), (right + 30, bottom + 35), (0, 255, 0), 2)
                    #cv2.rectangle(frame, (top, right), (top + bottom, bottom + left), (0, 255, 0), 2)
                    #cv2.ellipse(frame, frame.mean(axis=0), (100, 100), 0, 0, 360, (0, 255, 0), 2)
                    cv2.rectangle(frame, (left - 30, bottom + 35), (right + 30, bottom + 75), (0, 255, 0), -1)
                    cv2.putText(frame, split_name(name)[0], (left - 30, bottom + 60), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                    cv2.rectangle(frame, (left - 30, bottom + 75), (right + 30, bottom + 110), (0, 255, 0), -1)
                    cv2.putText(frame, split_name(name)[1], (left - 10, bottom + 95), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
            
            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) == ord('q'):
                break

            #time.sleep(.0333)

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # consider saving fr object to file to save load time (maybe pickle library)
    # ThreadedCamera('video.mp4')   # FAILED ATTEMPT AT THREADING
    # while True:                   # FAILED ATTEMPT AT THREADING
    #     time.sleep(1)             # FAILED ATTEMPT AT THREADING
    fr = FaceRecognition()    # reinstate if not using cProfile
    fr.run_recognition()  # reinstate if not using cProfile
    # cProfile.run('fr = FaceRecognition()')
    # cProfile.run('fr.run_recognition()')
    # dir_list = os.listdir('faces')
    # print(dir_list)
#pip install dlib==19.22
#pip install opencv_python
#pip install face_recognition

import face_recognition
import os, sys
import cv2
import numpy as np
import math
from moviepy.editor import VideoFileClip, AudioFileClip


def split_name(name):
    if name == 'Unknown (Unknown)':
        return 'Unknown', 'Unknown'
    parsed = name.split('.', 1)
    name_only = parsed[0]
    parsed_2 = parsed[1].split(' ')
    percentage_only = parsed_2[1]

    return name_only, percentage_only


def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


def add_audio_to_video(original_video_file, facial_video_file, output_file):
    original_video_clip = VideoFileClip(facial_video_file)
    facial_video_clip = VideoFileClip(original_video_file)
    audio_clip = facial_video_clip.audio
    final_clip = original_video_clip.set_audio(audio_clip)
    final_clip.write_videofile(output_file)

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


    def run_recognition(self, input_file, output_file):
        video_capture = cv2.VideoCapture(input_file)
        
        if not video_capture.isOpened():
            sys.exit('Video source not found . . .')

        # Get the frame width and height
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = video_capture.get(cv2.CAP_PROP_FPS)
        video_out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        #while True:
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret: break

            #frame = cv2.resize(frame, (0, 0), fx=(1/3), fy=(1/3))
            rgb_small_frame = np.ascontiguousarray(frame[:, :, ::-1])

            #find all the faces in the current frame
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

            # display and annotations (x, y, w, h)
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # top *= 3
                # right *= 3
                # bottom *= 3
                # left *= 3

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
            video_out.write(frame)

            if cv2.waitKey(1) == ord('q'):
                break

        video_out.release()

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()    # reinstate if not using cProfile
    video_input = 'Jacquemus_1080p_mini.mp4'
    video_only_output = 'Jacquemus_mini_output.mp4'
    synched_output = 'Jacquemus_mini_synched_output.mp4'
    fr.run_recognition(video_input, video_only_output)  # reinstate if not using cProfile
    add_audio_to_video(video_input, video_only_output, synched_output)
#pip install dlib==19.22
#pip install opencv_python
#pip install face_recognition

import face_recognition
import os, sys
import cv2
import numpy as np
import math
from moviepy.editor import VideoFileClip
from PIL import ImageFont, ImageDraw, Image
import pickle
import cProfile

# ----------------------------------------------------------
# This code was run on Python 3.10
# UNTESTED suggestions for newer Python versions are included below:
# Runing Dlib with Python newer than 3.10 requires a wheel file for Dlib
# Potential sources for dlib wheel files:
# pip install dlib-19.24.1-cp311-cp311-win_amd64.whl   # for Python 3.11
# pip install dlib-19.24.99-cp312-cp312-win_amd64.whl   # for Python 3.12
# ----------------------------------------------------------


def face_confidence(face_distance, face_match_threshold=0.6):
	range = (1.0 - face_match_threshold)
	linear_val = (1.0 - face_distance) / (range * 2.0)

	if face_distance > face_match_threshold:
		return round(linear_val * 100, 2)
	else:
		value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
		return round(value, 2)
	

def add_audio_to_video(original_video_file, facial_video_file, output_file):
	original_video_clip = VideoFileClip(facial_video_file)
	facial_video_clip = VideoFileClip(original_video_file)
	audio_clip = facial_video_clip.audio
	final_clip = original_video_clip.set_audio(audio_clip)
	final_clip.write_videofile(output_file)


class Encodings:
	def __init__(self):
		self.known_face_encodings = []
		self.known_face_names = []

	def save(self, path):
		with open(path, "wb") as f:
			pickle.dump(self, f)

	def load(self, path):
		with open(path, "rb") as f:
			obj = pickle.load(f)
			self.known_face_encodings = obj.known_face_encodings
			self.known_face_names = obj.known_face_names


class FaceRecognition:
	face_locations = []
	face_encodings = []
	#face_names = []
	known_face_encodings = []
	known_face_names = []
	process_current_frame = True
	frames = 0

	def __init__(self, path=None):
		if path is not None:
			try:
				self.load(path)
				print('Successfully loaded')
			except:
				print('Error loading file')
				self.encode_faces()
				self.save(path)
		else:
			self.encode_faces()
			self.save(path)

		def load_font(size):
			return ImageFont.truetype("Arial_Black.ttf", size)

		self.font_cache = {
			s: load_font(s) for s in (48, 24, 18, 75)
		}


	def encode_faces(self):
		file_names = os.listdir('Faces')
	#    print(file_names)
		for image_file_name in file_names:
			face_image = face_recognition.load_image_file(f'Faces/{image_file_name}')
			face_encodings = face_recognition.face_encodings(face_image)
			if face_encodings:
				face_encoding = face_encodings[0]
				self.known_face_encodings.append(face_encoding)
				self.known_face_names.append(image_file_name)
			else:
				print(f'ERROR-FILE: {image_file_name}')
					
		print(self.known_face_names)


	# These values tested as effective with threshold = 0.6
	def get_color_from_confidence(self, confidence):    # NOTE BGR not RGB
		# first returned color is for boxes and second is for True Font
		if confidence >= 97.0:
			return ((0, 255, 0), (0, 0, 0))
		elif 94.0 <= confidence < 97.0:
			return ((0, 255, 255), (0, 0, 0))
		elif 91.0 <= confidence < 94.0:
			return ((0, 126, 255), (0, 0, 0))
		elif 89.0 <= confidence < 91.0:
			return ((0, 0, 255), (0, 0, 0))
		elif 87.0 <= confidence < 89.0:
			return ((255, 255, 255), (0, 0, 0))
		else:
			return ((0, 0, 0), (255, 255, 255))

	def add_true_font(self, frame, text, pos, size, color):
		# Convert the image back to RGB (OpenCV uses BGR)
		cv2_im_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
 
		pil_im = Image.fromarray(cv2_im_rgb)

		draw = ImageDraw.Draw(pil_im)  

		font = self.font_cache[size]

		draw.text(pos, text, font=font, fill=color)  
		  
		return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

	
	def save(self, path):
		with open(path + "_encodings.dat", 'wb') as f:
			pickle.dump(self.known_face_encodings, f)
		with open(path + "_names.dat" , 'wb') as f:
			pickle.dump(self.known_face_names , f)


	def load(self, path):
		with open(path + "_encodings.dat", 'rb') as f:
			self.known_face_encodings = pickle.load(f)
		with open(path + "_names.dat", 'rb') as f:
			self.known_face_names = pickle.load(f)


	def run_recognition(self, input_file, output_file):
		video_capture = cv2.VideoCapture(input_file)
		
		if not video_capture.isOpened():
			sys.exit('Video source not found . . .')

		# Get the frame width and height
		frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
		frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

		fps = video_capture.get(cv2.CAP_PROP_FPS)
		video_out = cv2.VideoWriter(output_file, 
			cv2.VideoWriter_fourcc(*'mp4v'), 
			fps, 
			(frame_width, frame_height)
		)

		frame_count = 0
#		face_locations = []
		face_names = []
		face_confidences = []

		previous_locations = []
		previous_names = []
		previous_confidences = []

		#while True:
		while video_capture.isOpened():
			ret, frame = video_capture.read()
			if not ret: break

			frame_count += 1

			# ==========================================================
			# FACE DETECTION / RECOGNITION
			# ==========================================================

			if frame_count % process_every == 0:
				
				small_frame = cv2.resize(
					frame,
					(0, 0),
					fx = reduction_factor,	# Values <= 0.5 result in missed small faces
					fy = reduction_factor
				)

				rgb_small_frame = cv2.cvtColor(
					small_frame,
					cv2.COLOR_BGR2RGB
				)

				# ------------------------------------------------------
				# Detect faces
				# ------------------------------------------------------

				current_locations = face_recognition.face_locations(
					rgb_small_frame,
					model='HOG'
				)

				new_face_detected = len(current_locations) > len(previous_locations)

				if (
					frame_count == process_every
					or frame_count % (process_every * recognize_every) == 0
					or new_face_detected
				):

					#print("RECOGNIZING:", frame_count, "faces:", len(current_locations))

					self.face_encodings = face_recognition.face_encodings(
						rgb_small_frame,
						current_locations
					)

					current_names = []
					current_confidences = []

					known_encodings = self.known_face_encodings
					known_names = self.known_face_names

					for face_encoding in self.face_encodings:

						face_distances = face_recognition.face_distance(
							known_encodings,
							face_encoding
						)

						best_match_index = np.argmin(face_distances)

						name = 'Unknown'
						confidence = 0

						if face_distances[best_match_index] < 0.6:

							name = known_names[best_match_index]

							confidence = face_confidence(
								face_distances[best_match_index]
							)

						current_names.append(name)
						current_confidences.append(confidence)

					# These three lists are guaranteed to correspond.
					self.face_locations = current_locations
					face_names = current_names
					face_confidences = current_confidences

					# Save the recognized faces as our tracking reference.
					previous_locations = current_locations.copy()
					previous_names = current_names.copy()
					previous_confidences = current_confidences.copy()

				# ------------------------------------------------------
				# DETECTED A FACE, BUT DON'T RECOGNIZE IT
				# ------------------------------------------------------

				else:

					new_names = ['Unknown'] * len(current_locations)
					new_confidences = [0] * len(current_locations)

					used_previous = set()

					# Match each new face to ONE previous face.
					for i, current_location in enumerate(current_locations):

						current_top, current_right, current_bottom, current_left = current_location

						current_x = (current_left + current_right) / 2
						current_y = (current_top + current_bottom) / 2

						closest_index = None
						closest_distance = float('inf')

						for j, previous_location in enumerate(previous_locations):

							# Don't allow two current faces to use the same previous face.
							if j in used_previous:
								continue

							previous_top, previous_right, previous_bottom, previous_left = previous_location

							previous_x = (previous_left + previous_right) / 2
							previous_y = (previous_top + previous_bottom) / 2

							distance = (
								(current_x - previous_x) ** 2 +
								(current_y - previous_y) ** 2
							)

							if distance < closest_distance:

								closest_distance = distance
								closest_index = j

						# 60 pixels is measured on the reduced-size image
						if (
							closest_index is not None
							and closest_distance < (60 ** 2)
						):

							new_names[i] = previous_names[closest_index]
							new_confidences[i] = previous_confidences[closest_index]

							used_previous.add(closest_index)

					# --------------------------------------------------
					# The CURRENT locations get the CURRENT names.
					# --------------------------------------------------

					self.face_locations = current_locations
					face_names = new_names
					face_confidences = new_confidences

					# --------------------------------------------------
					# IMPORTANT:
					#
					# Move the tracking reference forward.
					#
					# We keep the identity but update its location
					# This means a moving person doesn't have to remain
					# within 60 pixels of the location from recognize_every frames ago
					# --------------------------------------------------

					previous_locations = current_locations.copy()
					previous_names = new_names.copy()
					previous_confidences = new_confidences.copy()

			# ----------------------------------------------------------
			# DETECTION WITHOUT RECOGNITION
			# ----------------------------------------------------------

			else:

				new_face_names = []
				new_face_confidences = []

				for current_location in self.face_locations:

					current_top, current_right, current_bottom, current_left = current_location

					current_x = (current_left + current_right) / 2
					current_y = (current_top + current_bottom) / 2

					closest_index = None
					closest_distance = float('inf')

					for i, previous_location in enumerate(previous_locations):

						previous_top, previous_right, previous_bottom, previous_left = previous_location

						previous_x = (previous_left + previous_right) / 2
						previous_y = (previous_top + previous_bottom) / 2

						distance = (
							(current_x - previous_x) ** 2 +
							(current_y - previous_y) ** 2
						)

						if distance < closest_distance:

							closest_distance = distance
							closest_index = i

					if (
						closest_index is not None
						and closest_distance < 60 ** 2
					):

						new_face_names.append(
							previous_names[closest_index]
						)

						new_face_confidences.append(
							previous_confidences[closest_index]
						)

					else:

						new_face_names.append('Unknown')
						new_face_confidences.append(0)

				face_names = new_face_names
				face_confidences = new_face_confidences

			overlay = frame.copy()
			font = cv2.FONT_HERSHEY_SIMPLEX  # determines cv2.getTextSize which drives name/confidence box size
			text_box_alpha = 0.5

			#font_transparency = 128
			for (top, right, bottom, left), name, confidence in zip(self.face_locations, face_names, face_confidences):
				top = round(top * (1/reduction_factor))
				right = round(right * (1/reduction_factor))
				bottom = round(bottom * (1/reduction_factor))
				left = round(left * (1/reduction_factor))

				if name != 'Unknown':
					name = name[:-4]
				face_box_width = right - left
				confidence_str = str(confidence) + '%'
				# conditional below eliminates lower confidence boxes - match with get_color_from_confidence
				if confidence >= 91.0:
					box_color = self.get_color_from_confidence(confidence)[0]
					ttf_color = self.get_color_from_confidence(confidence)[1]

					# Create box that encircles identified face with transparency of main_box_alpha
					cv2.rectangle(
						overlay, 
						(left - 30, top - 75), 
						(right + 30, bottom + 35), 
						box_color, 
						2
					)
					#image_new_main = cv2.addWeighted(overlay, main_box_alpha, frame, 1 - main_box_alpha, 0)

					textsize_name = cv2.getTextSize(name, font, 1.7, 2)[0]
					text_length_name = textsize_name[0]

					# Create box for name text with transparency of text_box_alpha
					cv2.rectangle(
						overlay, 
						(left + int(face_box_width/2) - int(text_length_name/2) - 10, 
	   					bottom + 35), 
						(right - int(face_box_width/2) + int(text_length_name/2) + 10, 
						bottom + 85), 
						box_color, 
						-1
					)
					name_loc = (left + int(face_box_width/2) - int(text_length_name/2), bottom + 20)    # bottom currently hard-coded

					textsize_confidence = cv2.getTextSize(confidence_str, font, 0.8, 2)[0]
					text_length_confidence = textsize_confidence[0]

					# Create box that encircles confidence with transparency of confidence_box_alpha
					cv2.rectangle(
						overlay, 
						(left + int(face_box_width/2) - int(text_length_confidence/2) - 10, bottom + 85), 
						(right - int(face_box_width/2) + int(text_length_confidence/2) + 10, bottom + 105), 
						box_color, 
						-1
					)
					confidence_loc = (
						left + int(face_box_width/2) - int(text_length_confidence/2), 
						bottom + 75
					)
					overlay = self.add_true_font(overlay, name, name_loc, 48, ttf_color)
					overlay = self.add_true_font(overlay, confidence_str, confidence_loc, 24, ttf_color)

			# Blend the overlay wit the original image
			frame = cv2.addWeighted(frame, 1 - text_box_alpha, overlay, text_box_alpha, 0)

			cv2.imshow('Face Recognition', frame)
			video_out.write(frame)

			if cv2.waitKey(1) == ord('q'):
				break

		video_out.release()

		video_capture.release()
		cv2.destroyAllWindows()


if __name__ == '__main__':
	file_name = 'saved_face'
	fr = FaceRecognition(path=file_name)    # reinstate if not using cProfile

	# ----------------------------------------------------------
	# GLOBAL OPTIMIZATION VARIABLES: values of one = processing
	# each video frame full size and recognizing faces in each frame
	# ----------------------------------------------------------
	process_every = 3	#Set to 1 to process every frame (3 creates no noticeable lag)
	recognize_every = 6	#Set at 1 to recognize every frame (12 creates noticeable lag)
	reduction_factor = 5/8	# Values <= 0.5 result in missed small faces

	video_input = '3_SEC_TEST_INPUT_VIDEO.mp4'
	video_only_output = '3_SEC_TEST_INPUT_VIDEO_0.6_above_91_3-6_5-8_output.mp4'
	synched_output = '3_SEC_TEST_INPUT_VIDEO_0.6_above_91_3-6_5-8_synched_output.mp4'
	# run next line only for timings only
	#cProfile.run("fr.run_recognition(video_input, video_only_output)", sort='cumulative')
	fr.run_recognition(video_input, video_only_output)  # reinstate if not using cProfile
	fr.save(file_name)
	add_audio_to_video(video_input, video_only_output, synched_output)
Over a few years I developed a Python script that takes in a video then using a training file of face photographs creates embeddings of the training data set
and applies Dlib facial recognition to the input video resulting in an output video with attractive facial IDs in boxes including confidence percentage boxes.

I run this on Python 3.10 and newer version of Python require a Dlib wheel (see comments in the code for wheel suggestions for 3.11 and 3.12)

Recently I have optimized the code because when processing every frame of video for facial recognition it is very slow. In the main section see speed
optimization adjustments that perform three main tasks:

  reduction_factor scales down the video frames so there is less video to process when performing the facial recognition function
  process_every limits the number of frames that are reviewed to find faces
  recognize_every limits how often the faces are matched in the processed frames
  
These adjustments have a profound effect on the speed of the program. I have included some composite videos of test using the provided three second test video.

  As seen in 5-8_Testing_Diff_Freq.mp4:
    When reduction_factor was set to 5/8 and both process_every and recognize_every were set to one so that all frames where processed it took 182 seconds,
    When reduction_factor was set to 5/8 and process_every = 2 and recognize_every = 4 it took 63 seconds, and
    When reduction_factor was set to 5/8 and process_every = 3 and recognize_every = 6 it took 43 seconds

A reduction_factor of 5/8 resulted in good quality, but as seen in 1-2_Testing_Diff_Freq.mp4 taking reduction_factor down to 1/2 resulted in poor quality
due to the program's difficulty in identifying small faces in the video frames.

Nothing groundbreaking here, but I have done a lot of nit-picky work to get the program to run faster and the facial IDs to look attractive in the output video.

import argparse

import kornia_io as K
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def my_app(args):
  # create the video capture object
  cap = K.CameraStream.create(K.CameraStreamBackend.OPENCV)

  # create the video writer
  writer = K.VideoStreamWriter(
    args.output_file, args.fps, cap.resolution)

  # create the visualizer object
  viz = K.Visualizer()

  # use mediapipe to detect hands
  with mp_hands.Hands(
      model_complexity=0,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as hands:
    while cap.is_opened():
      image: K.Image = cap.get()
      if not image.valid:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # detect the hands with mediapipe
      image_np = image.byte().to_numpy()  # media pipe needs numpy/uint8
      results = hands.process(image_np)  # i think it flips the image

      # Draw the hand annotations on the image.
      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(
              image_np,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())

      # visualize with visdom and save frame to disk
      image_out = K.Image.from_numpy(image_np).hflip()
      viz.show_image('frame_out', image_out)
      writer.append(image_out)

  cap.close()
  writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face and Landmark Detection')
    parser.add_argument('--output_file', required=True, type=str, help='the file to save in disk.')
    parser.add_argument('--fps', default=30.0, type=float, help='the frames per second.')
    args = parser.parse_args()
    my_app(args)
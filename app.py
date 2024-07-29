import cv2
import mediapipe as mp
import streamlit as st
import numpy as np

def main():
    st.title('Cilt Rengi Tespit Uygulaması')

    frame_placeholder = st.empty()
    stop_button = st.button("Stop")

    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    cap = cv2.VideoCapture(-1)

    def get_skin_color(image1, landmarks):
        height, width, _ = image1.shape
        regions = [
            landmarks.landmark[234],  # left cheek
            landmarks.landmark[454],  # right cheek
            landmarks.landmark[10],   # forehead
        ]
        skin_colors = []
        for region in regions:
            x = int(region.x * width)
            y = int(region.y * height)
            color = image1[y-5:y+5, x-5:x+5]  # Take a 10x10 region around the point
            avg_color = np.mean(color, axis=(0, 1))
            skin_colors.append(avg_color)
        return np.mean(skin_colors, axis=0)

    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                st.write("Video Capture Ended")
                break

            # Flip the image horizontally for a later selfie-view display
            # Convert the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # Adjust white balance
            result = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            avg_a = np.average(result[:, :, 1])
            avg_b = np.average(result[:, :, 2])
            result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
            result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
            image = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

            # Equalize histogram
            img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

            image.flags.writeable = False
            results = face_mesh.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)
                    skin_color = get_skin_color(image, face_landmarks)
                    # skin_color_bgr = (int(skin_color[0]), int(skin_color[1]), int(skin_color[2]))
                    # st.write(f"Average Skin Color (BGR): {skin_color_bgr}")
                    cv2.rectangle(image, (10, 10), (200, 200),
                                  (int(skin_color[0]), int(skin_color[1]), int(skin_color[2])), -1)
            frame_placeholder.image(image, channels="BGR")

            if stop_button:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

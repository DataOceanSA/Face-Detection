import streamlit as st
import cv2
import os

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default .xml")

def save_image_with_faces(frame, faces):
    # Create a folder to store the images if it doesn't exist
    if not os.path.exists("saved_images"):
        os.makedirs("saved_images")
    
    # Save each image with detected faces
    for i, (x, y, w, h) in enumerate(faces):
        face_img = frame[y:y+h, x:x+w]
        cv2.imwrite(f"saved_images/face_{i}.jpg", face_img)


def main():
    st.set_page_config(page_title="Streamlit WebCam App")
    st.title("Webcam Display Streamlit App")
    st.caption("Powered by OpenCV, Streamlit")
    
    # Add instructions
    st.markdown("## Instructions:")
    st.write("1. Press the 'Stop' button to stop capturing video.")
    st.write("2. The app detects faces in real-time using your webcam.")
    st.write("3. Press the 'Save Faces' button to save images with detected faces.")
    
    # Add color picker
    selected_color = st.color_picker("Choose color for rectangles", "#00FF00")
    selected_color_bgr = tuple(int(selected_color[i:i+2], 16) for i in (1, 3, 5))
    min_neighbors = st.slider("minNeighbors", min_value=1, max_value=10, value=5)
    scale_factor = st.slider("scaleFactor", min_value=1.1, max_value=2.0, step=0.1, value=1.3)
    
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop_button_pressed = st.button("Stop")
    save_button_pressed = st.button("Save Faces")
    
    while cap.isOpened() and not stop_button_pressed:
        ret, frame = cap.read()
        if not ret:
            st.write("Error: Unable to capture frame from the webcam.")
            continue  # Skip processing empty frames
        
        # Convert the frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect the faces using the face cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        # Draw rectangles around the detected faces with the selected color
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), selected_color_bgr, 2)
        
        if not ret:
            st.write("Video Capture Ended")
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")
        
        if save_button_pressed:
            save_image_with_faces(frame, faces)
            st.write("Images with detected faces saved successfully!")
            save_button_pressed = False
        
        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
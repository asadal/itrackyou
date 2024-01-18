# import torch
# import torchvision

# print(torch.__version__)
# print(torchvision.__version__)

import numpy as np
import supervision as sv
from ultralytics import YOLO
import os
import streamlit as st
from tempfile import NamedTemporaryFile
import cv2
from collections import Counter

model = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()
tracked_objects = {}

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    global tracked_objects
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
        label = results.names[class_id]
        tracked_objects[tracker_id] = label
    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections)
    return trace_annotator.annotate(annotated_frame, detections=detections)

def save_uploaded_file(uploaded_file):
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        return temp_file.name

def string_to_byte(filepath):
    with open(filepath, 'rb') as f:
        file_path = f.read()
    return file_path

def delete_original_video(video_file_path):
    if os.path.exists(video_file_path):
        os.remove(video_file_path)
    st.session_state.video_processed = False

def count_objects(labels=list):
    counts = dict(Counter(labels))
    return counts

def display_tracked_objects():
    global tracked_objects
    counts = Counter(tracked_objects.values())
    st.write(counts)
    tracked_objects = {}

def app():
    st.set_page_config(
        page_title="I Track You",
        page_icon="https://raw.githubusercontent.com/asadal/itrackyou/main/images/track_pictogram.png"
    )
    st.image(
        "https://raw.githubusercontent.com/asadal/itrackyou/main/images/track_pictogram.png",
        width=150
    )
    st.title("I Track You")
    
    track_menu = ['Track Image', 'Track Video']

    with st.sidebar:
        menu_select = st.radio(
            "I wanna track...",
            track_menu,
            key="radio"
        )
    
    if menu_select == track_menu[0]:
        st.header("Track Image")
        st.markdown("### Just **upload** image. That's it!")
        st.markdown("If not working, refresh page. 🔄")
        
        image_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        
        if 'image_processed' not in st.session_state:
            st.session_state.image_processed = False
        
        if image_file is not None:
            st.session_state.image_processed = False
            st.image(image_file)
            image_file_path = save_uploaded_file(image_file)
            if st.button("Track and Download"):
                with st.spinner("Tracking objects..."):
                    image = cv2.imread(image_file_path)
                    results = model(image)[0]
                    detections = sv.Detections.from_ultralytics(results)
                    bounding_box_annotator = sv.BoundingBoxAnnotator()
                    label_annotator = sv.LabelAnnotator()
                    labels = [
                        model.model.names[class_id]
                        for class_id
                        in detections.class_id
                    ]
                    counts = count_objects(labels)
                    annotated_image = bounding_box_annotator.annotate(
                        scene=image, detections=detections)
                    annotated_image = label_annotator.annotate(
                        scene=annotated_image, detections=detections, labels=labels)
                st.write("Tracking complete!")
                st.image(annotated_image)
                st.write(counts)
                annotated_image_bytes = cv2.imencode('.png', annotated_image)[1].tobytes()
                st.download_button(
                    label="Download Output Image",
                    data=annotated_image_bytes,
                    file_name="output_" + image_file.name,
                    on_click=delete_original_video(image_file_path))
                if image_file_path is None:
                    st.write("Original image file was deleted.")
                st.session_state.image_processed = True
                st.session_state.image_file = None
    else:
        st.markdown("### Just **upload** video. That's it!")
        st.markdown("If not working, refresh page. 🔄")

        video_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])

        if 'video_processed' not in st.session_state:
            st.session_state.video_processed = False

        if video_file is not None:
            st.session_state.video_processed = False
            st.video(video_file)
            video_file_path = save_uploaded_file(video_file)
            if st.button("Track and Download"):
                with st.spinner("Tracking objects..."):
                    cap = cv2.VideoCapture(video_file_path)
                    frame_width = int(cap.get(3))
                    frame_height = int(cap.get(4))
                    out = cv2.VideoWriter(
                        "output_video.mp4",
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        30,
                        (frame_width, frame_height)
                    )
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        annotated_frame = callback(frame, 0)
                        out.write(annotated_frame)
                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()
                st.write("Tracking complete!")
                st.video("output_video.mp4")
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                st.download_button(
                    label="Download Output Video",
                    data=string_to_byte("output_video.mp4"),
                    file_name="output_video.mp4",
                    on_click=delete_original_video(video_file_path))
                if video_file_path is None:
                    st.write("Original video file was deleted.")
                st.session_state.video_processed = True
                st.session_state.video_file = None
                
if __name__ == "__main__":
    app()

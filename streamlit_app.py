import numpy as np
import supervision as sv
from ultralytics import YOLO
import os
import streamlit as st
from tempfile import NamedTemporaryFile

model = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
# 하단 원 모양(ellipse)으로 바꾸기
# ellipse_annotator = sv.EllipseAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    annotated_frame = box_annotator.annotate(
        frame.copy(), detections=detections)
    # annotated_frame = ellipse_annotator.annotate(
    #     frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels)
    return trace_annotator.annotate(
        annotated_frame, detections=detections)

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

###########################

def app():
    # Set page title and icon
    st.set_page_config(
        page_title="I Track You",
        page_icon="https://static-00.iconduck.com/assets.00/radar-icon-2048x2048-parhwoy9.png"
    )
    # Featured image
    st.image(
        "https://static-00.iconduck.com/assets.00/radar-icon-2048x2048-parhwoy9.png",
        width=150
    )
    # Main title and description
    st.title("I Track You")
    st.markdown("### Just **upload** video. That's it!")
    st.markdown("If not working, Refresh page. ⟳")

    
    video_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])

    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False

    if video_file is not None:
        st.session_state.video_processed = False
        st.video(video_file)
        video_file_path = save_uploaded_file(video_file)
        output_file_name = "output_" + video_file.name
        if st.button("Track and Download"):
            with st.spinner("Tracking objects..."):
                with NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                    tmp_file.write(video_file.getvalue())
                    file_path = tmp_file.name
                    output_file_path = os.path.splitext(file_path)[0] + "_output.mp4"
                    sv.process_video(
                        source_path=file_path,
                        target_path=output_file_path,
                        callback=callback
                    )
            st.write("Tracking complete!")
            output_data = string_to_byte(output_file_path)
            st.video(
                data=output_data,
                format='video/mp4')
            st.write("Can't see the video properly? Download it instead.")
            st.download_button(
                label="Download Output Video",
                data=output_data,
                file_name=output_file_name,
                on_click=delete_original_video(video_file_path))
            st.session_state.video_processed = True
            st.session_state.video_file = None

        
if __name__ == "__main__":
    app()

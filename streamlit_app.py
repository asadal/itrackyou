import numpy as np
import supervision as sv
from ultralytics import YOLO
import os
import streamlit as st
import re
import tempfile as tf
from tempfile import NamedTemporaryFile
import shutil

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

def string_to_byte(filepath):
    with open(filepath, 'rb') as f:
        file_path = f.read()
    return file_path

###########################

def app():
    # Set page title and icon
    st.set_page_config(
        page_title="I Trach You",
        page_icon="https://static-00.iconduck.com/assets.00/radar-icon-2048x2048-parhwoy9.png"
    )
    # Featured image
    st.image(
        "https://static-00.iconduck.com/assets.00/radar-icon-2048x2048-parhwoy9.png",
        width=150
    )
    # Main title and description
    st.title("I Track You")
    st.markdown("Just **upload** video. That's it!")

    video_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])

    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False

    if video_file is not None:
        if not st.session_state.video_processed:
            st.video(video_file)
            output_file_name = "output_" + video_file.name
            # print("비디오 파일 형식은 : ", type(video_file))
            with st.spinner("Tracking objects..."):
                with NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                    tmp_file.write(video_file.getvalue())
                    file_path = tmp_file.name
                    # print("파일 경로 : ", file_path)
                    output_file_path = os.path.splitext(file_path)[0] + "_output.mp4"
                    sv.process_video(
                        source_path=file_path,
                        target_path=output_file_path,
                        callback=callback
                        )
            st.write("Tracking complete!")
            if os.path.exists(output_file_path):
                print(f"File exists at {output_file_path}")
            else:
                print(f"File does not exist at {output_file_path}")
            output_data = string_to_byte(output_file_path)
            
            # 추적된 동영상 보기(제대로 안 보일 수 있음)
            st.video(
                data=output_data, 
                format='video/mp4')
            st.write("Can't see the video properly? Download it instead.")
            st.download_button(
                label="Download Output Video", 
                data=output_data,
                file_name=output_file_name)
            st.session_state.video_processed = True

if __name__ == "__main__":
    app()

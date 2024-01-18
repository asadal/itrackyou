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
# 하단 원 모양(ellipse)으로 바꾸기
# ellipse_annotator = sv.EllipseAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

# 전역 변수로 각 추적 ID에 대한 레이블을 저장할 딕셔너리 선언
tracked_objects = {}

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    global tracked_objects  # 전역 변수 사용을 위한 선언

    # 모델을 사용하여 프레임에서 객체 검출
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    # 추적된 객체들의 ID와 레이블을 기록
    for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
        label = results.names[class_id]
        tracked_objects[tracker_id] = label

    # 프레임에 바운딩 박스와 레이블을 주석으로 추가
    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections)

    # 추적 경로를 프레임에 주석으로 추가
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
    
# 추가: 동영상 처리 후에 추적된 객체 종류별로 고유한 추적 ID의 수를 계산 및 출력
def display_tracked_objects():
    global tracked_objects
    counts = Counter(tracked_objects.values())
    st.write(counts)
    tracked_objects = {}  # 딕셔너리 초기화

# 사진 속 사물 개수 세기
def count_objects(labels=list):
    counts = dict(Counter(labels))
    return(counts)

###########################

def app():
    # Set page title and icon
    st.set_page_config(
        page_title="I Track You",
        page_icon="https://raw.githubusercontent.com/asadal/itrackyou/main/images/track_pictogram.png"
    )
    # Featured image
    st.image(
        "https://raw.githubusercontent.com/asadal/itrackyou/main/images/track_pictogram.png",
        width=150
    )
    # Main title and description
    st.title("I Track You")
    
    track_menu = ['Track Image', 'Track Video']

    with st.sidebar:
        menu_select = st.radio(
            "I wanna...",
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
                    
                    # 사진 속 사물들 개수 세기
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
            # else:
            #     st.write("Upload a new image to track.")
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
                st.video(output_data)
                display_tracked_objects()  # 추가: 결과 출력 함수 호출
                st.write("Can't see the video properly? Download it instead.")
                st.download_button(
                    label="Download Output Video",
                    data=output_data,
                    file_name=output_file_name,
                    on_click=delete_original_video(video_file_path))
                if video_file_path is None:
                    st.write("Original video file was deleted.")
                st.session_state.video_processed = True
                st.session_state.video_file = None
            # else:
            #     st.write("Upload a new video to track.")
        
if __name__ == "__main__":
    app()

import numpy as np
import supervision as sv
from ultralytics import YOLO
import os
import streamlit as st
from tempfile import NamedTemporaryFile
import cv2

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
    
# def make_result_dict(img):
#     results = model(img)[0].names
# # 딕셔너리의 값들의 개수를 세는 방법
#     counts = {}
#     for value in results.values():
#         counts[value] = counts.get(value, 0) + 1
#     return counts

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
        # if 'image_file' not in st.session_state:
        #     st.session_state.image_file = None
        
        if image_file is not None:
            st.session_state.image_processed = False
            # if not st.session_state.image_processed or st.session_state.image_file != image_file:
            # st.session_state.image_file = image_file
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

                    annotated_image = bounding_box_annotator.annotate(
                        scene=image, detections=detections)
                    annotated_image = label_annotator.annotate(
                        scene=annotated_image, detections=detections, labels=labels)
                st.write("Tracking complete!")
                st.image(annotated_image)
                # print("이미지 포맷 : " + annotated_image.format)
                annotated_image_bytes = cv2.imencode('.png', annotated_image)[1].tobytes()
                # results = make_result_dict(image)
                # for key, value in results.items():
                #     st.write(f"{key} : {value}")
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
        # if 'video_file' not in st.session_state:
        #     st.session_state.video_file = None

        if video_file is not None:
            st.session_state.video_processed = False
            # if not st.session_state.video_processed or st.session_state.video_file != video_file:
            # st.session_state.video_file = video_file
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
                if video_file_path is None:
                    st.write("Original video file was deleted.")
                st.session_state.video_processed = True
                st.session_state.video_file = None
            # else:
            #     st.write("Upload a new video to track.")
        
if __name__ == "__main__":
    app()

# 출처 : 
# - https://supervision.roboflow.com/how_to/track_objects/
# - https://supervision.roboflow.com/trackers/ 
# - https://github.com/roboflow/supervision

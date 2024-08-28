import cv2
import numpy as np
import os
from ultralytics import YOLO

model = YOLO('t_bag_box.pt')

video_path = r'D:\vChanel\truck\Bale_Count.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
output_path = 'bale_output.mp4'  # name for the video to save
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

zone = [(1040,21), (1139,19), (1141,1026), (1038,1028)]


object_count = 0
previous_count = 0


recent_frames = []

# Create an output directory if it doesn't exist
output_folder = 'bale_output'
os.makedirs(output_folder, exist_ok=True)

# Function to check if a frame is recent
def is_frame_recent(frame_index, recent_frames):
    for i in recent_frames:
        if frame_index - i < 30:
            return True
    return False

frame_index = 0  # Initialize the frame index

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    results = model(frame)


    for result in results:
        for box in result.boxes:
            # Filter by confidence
            if box.conf >= 0.60:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)


                bbox_width = x2 - x1
                bbox_height = y2 - y1

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # Calculate and draw the centroid
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)
                cv2.circle(frame, (centroid_x, centroid_y), 5, (255, 255, 0), -1)
                size_text = f"W: {bbox_width} H: {bbox_height}"
                cv2.putText(frame, size_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if bbox_width > 50 and bbox_height > 50:

                    if cv2.pointPolygonTest(np.array(zone, np.int32), (centroid_x, centroid_y), False) >= 0:
                        if not is_frame_recent(frame_index, recent_frames):

                            if  bbox_height > 180:
                                object_count += 2
                            else:
                                object_count += 1

                            recent_frames.append(frame_index)

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 5)


    cv2.polylines(frame, [np.array(zone, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)


    cv2.putText(frame, f"Count: {object_count}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)


    if object_count > previous_count:
        frame_filename = os.path.join(output_folder, f"frame_{frame_index}.jpg")
        cv2.imwrite(frame_filename, frame)
        previous_count = object_count


    out.write(frame)

    disp = cv2.resize(frame, (800, 800))

    cv2.imshow('Object Detection',  disp)

    frame_index += 1


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

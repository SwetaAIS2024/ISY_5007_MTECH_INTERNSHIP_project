import cv2
import numpy as np

# Global variables to store points and polygons
drawing = False
current_polygon = []
polygons = []

def draw_polygon(event, x, y, flags, param):
    global drawing, current_polygon, polygons

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing a polygon
        drawing = True
        current_polygon.append((x, y))

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # Update the current polygon as the user moves the mouse
        pass

    elif event == cv2.EVENT_LBUTTONUP:
        # Add the point to the current polygon
        current_polygon.append((x, y))

def annotate_zones(video_path):
    global drawing, current_polygon, polygons

    # Open the video and grab the first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return []

    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]

    # Set up the OpenCV window and mouse callback
    cv2.namedWindow("Annotate Zones")
    cv2.setMouseCallback("Annotate Zones", draw_polygon)

    while True:
        temp_frame = frame.copy()

        # Draw existing polygons
        for polygon in polygons:
            cv2.polylines(temp_frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw the current polygon being created
        if len(current_polygon) > 1:
            cv2.polylines(temp_frame, [np.array(current_polygon, dtype=np.int32)], isClosed=False, color=(0, 0, 255), thickness=2)

        cv2.imshow("Annotate Zones", temp_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Quit annotation
            # If the current polygon is incomplete, add it to the list
            if len(current_polygon) > 2:
                polygons.append(np.array(current_polygon, dtype=np.int32))
            break
        elif key == ord("n"):  # Finish the current polygon and start a new one
            if len(current_polygon) > 2:
                polygons.append(np.array(current_polygon, dtype=np.int32))
            current_polygon = []
        elif key == ord("r"):  # Reset all annotations
            polygons = []
            current_polygon = []

    cv2.destroyAllWindows()
    cap.release()

    # Normalize polygons
    normalized_polygons = []
    for polygon in polygons:
        normalized_polygon = polygon.astype(float)
        normalized_polygon[:, 0] /= frame_width  # Normalize x-coordinates
        normalized_polygon[:, 1] /= frame_height  # Normalize y-coordinates
        normalized_polygons.append(normalized_polygon)

    return normalized_polygons
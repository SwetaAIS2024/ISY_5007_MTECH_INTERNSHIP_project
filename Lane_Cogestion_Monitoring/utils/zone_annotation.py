import cv2
import numpy as np

# Global variables to store points and polygons
drawing = False
current_polygon = []
polygons = []

def draw_polygon(event, x, y, flags, param):
    """Mouse callback function to draw polygons."""
    global drawing, current_polygon, polygons

    if event == cv2.EVENT_LBUTTONDOWN:
        # Add a point to the current polygon
        current_polygon.append((x, y))
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # Show preview line
        param['preview'] = (x, y)

def annotate_zones(frame):
    """Annotate zones on the given frame."""
    global drawing, current_polygon, polygons

    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]

    # Reset global variables
    drawing = False
    current_polygon = []
    polygons = []

    # Set up the OpenCV window and mouse callback
    cv2.namedWindow("Annotate Zones")
    preview = {'preview': None}
    cv2.setMouseCallback("Annotate Zones", draw_polygon, preview)

    print("\nZone Annotation Instructions:")
    print("- Left click: Add point")
    print("- 'n': Complete current polygon and start a new one")
    print("- 'r': Reset all polygons")
    print("- 's': Save and exit")
    print("- 'q': Quit without saving\n")

    while True:
        temp_frame = frame.copy()

        # Draw existing polygons
        for polygon in polygons:
            cv2.polylines(temp_frame, [polygon], True, (0, 255, 0), 2)

        # Draw the current polygon
        if current_polygon:
            points = np.array(current_polygon, dtype=np.int32)
            cv2.polylines(temp_frame, [points], False, (0, 0, 255), 2)

            # Draw preview line
            if preview['preview']:
                last_point = np.array(current_polygon[-1])
                preview_point = np.array(preview['preview'])
                cv2.line(temp_frame, tuple(last_point), tuple(preview_point), (255, 0, 0), 1)

        cv2.imshow("Annotate Zones", temp_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Save and exit
            if current_polygon and len(current_polygon) > 2:
                polygons.append(np.array(current_polygon, dtype=np.int32))
            if polygons:
                normalized_polygons = []
                for polygon in polygons:
                    norm_polygon = polygon.astype(float)
                    norm_polygon[:, 0] /= frame_width
                    norm_polygon[:, 1] /= frame_height
                    normalized_polygons.append(norm_polygon)
                print(f"Saved {len(polygons)} polygons.")
                cv2.destroyAllWindows()
                return normalized_polygons
            break
        elif key == ord('q'):  # Quit without saving
            break
        elif key == ord('n'):  # Complete current polygon and start a new one
            if len(current_polygon) > 2:
                polygons.append(np.array(current_polygon, dtype=np.int32))
            current_polygon = []
            drawing = False
        elif key == ord('r'):  # Reset all polygons
            polygons = []
            current_polygon = []
            drawing = False

    cv2.destroyAllWindows()
    return []
import requests
import datetime
import os

# 👉 Replace this with your actual API key
API_KEY = os.getenv('LTA_API_KEY')

# Create output folder
os.makedirs("lta_intersection_images", exist_ok=True)

# API endpoint and headers
url = 'https://api.data.gov.sg/v1/transport/traffic-images'
headers = {'AccountKey': API_KEY}

# Call the API
response = requests.get(url, headers=headers)
data = response.json()

# # Known camera IDs at intersections (manually curated)
# intersection_camera_ids = [
#     "2701",  # Example: PIE slip road
#     "2702",  # Example: Thomson Rd intersection
#     "1001",  # Example: Bukit Timah
#     # Add more camera IDs after inspecting the metadata / testing visually
# ]

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Save images only from intersection cameras
image_count = 0
max_images = 100000
for cam in data['items'][0]['cameras']:
    if image_count >= max_images:
        break
    cam_id = cam['camera_id']
    #if cam_id in intersection_camera_ids:
    img_url = cam['image']
    filename = f"lta_intersection_images/{cam_id}_{timestamp}.jpg"
    try:
    # Use requests.get with User-Agent header to avoid 403
        img_response = requests.get(img_url, headers={"User-Agent": "Mozilla/5.0"})
        if img_response.status_code == 200:
               with open(filename, 'wb') as f:
                       f.write(img_response.content)
               print(f"Saved {filename}")
               image_count += 1
        else:
               print(f"Failed to download {img_url}: {img_response.status_code}")
    except Exception as e:
        print(f"Error downloading {img_url}: {e}")
import os
from PIL import Image, ImageFilter

# Specify the folder path where the images are located
folder_path = '/home/redpc/Documents/Code/InsightFace-REST/src/api_trt/images_data/old.img_BU/27122019/10234/image'

# Create a new folder to store the filtered images
filtered_folder_path = os.path.join(folder_path, 'filtered')
os.makedirs(filtered_folder_path, exist_ok=True)

# Process frame
def processing(image_path, mode='LAB'):
	image = Image.open(image_path)
    if (mode == 'HSV'):    # HSV color space
        hsv_image_path = cv2.cvtColor(image_path, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image_path)
        h = clahe_h.apply(h)
        s = clahe_s.apply(s)
        v = clahe_v.apply(v)
        processed_image_path = cv2.merge([h, s, v])
        processed_image_path = cv2.cvtColor(processed_image_path, cv2.COLOR_HSV2BGR)
    elif (mode == 'LAB'):    # LAB color space
        lab_frame = cv2.cvtColor(image_path, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab_image_path)
        l = clahe_l.apply(l)
        processed_image_path = cv2.merge([l, a, b])
        processed_image_path = cv2.cvtColor(processed_image_path, cv2.COLOR_LAB2BGR)
    

   	processed_image_path.save(os.path.join(filtered_folder_path, os.path.basename(image_path)))    
    return processed_image_path


# Iterate through all files in the folder and apply the filter to image files
for file_name in os.listdir(folder_path):
    if file_name.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(folder_path, file_name)
        processing(image_path, mode ='LAB')  # Replace ImageFilter.BLUR with the desired filter type

print("Filtering complete.")


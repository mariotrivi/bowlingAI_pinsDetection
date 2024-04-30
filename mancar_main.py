from ultralytics import YOLO
import cv2
import numpy as np
import os
import shutil 
import glob
import json
from DepthAnything.metric_depth.depth_single_image import Estimator as DA_estimator

def analyze_depth(img_depth, np_mask):
    """
    Calculates the average depth of the object in the depth image specified by the mask.
    In debug mode, it also saves the depth image and the masked depth image.

    Parameters:
    - img_depth: The depth image where each pixel represents the estimated depth.
    - np_mask: A mask indicating the pixels belonging to the object of interest.
    Returns:
    - The average depth of the object.
    """
    # Aplicar la máscara a la imagen de profundidad
    object_pixels = img_depth[np_mask.astype(bool)]
    
    # Calcular la profundidad media
    mean_depth = np.mean(object_pixels)

    return mean_depth

def clear_debug_folder(debug_folder="debug_images"):
    """
    Clears the contents of the debug folder, if it exists, to start with an empty folder for each run.

    Parameters:
    - debug_folder: The folder where debug images will be saved.
    """
    if os.path.exists(debug_folder):
        shutil.rmtree(debug_folder)
    os.makedirs(debug_folder)

def resize_image(image, max_dim=640):
    """
    Resizes the provided image so that its maximum dimension is equal to max_dim pixels,
    maintaining the aspect ratio and adding padding to make the image square.

    Parameters:
    - image: The original image as a NumPy array.
    - max_dim: The maximum dimension of the resized image.

    Returns:
    - The resized and squared image with padding, if necessary.
    """
    h, w = image.shape[:2]
    scale = max_dim / max(h, w)
    scale = scale if scale < 1 else 1
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padding = [(max_dim - new_h) // 2, max_dim - new_h - (max_dim - new_h) // 2,
               (max_dim - new_w) // 2, max_dim - new_w - (max_dim - new_w) // 2]
    padded_image = cv2.copyMakeBorder(resized_image, *padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

def crop_object(image, mask, margin=20):
    """
    Crops the image to include only the object of interest based on the provided mask, with an additional margin.

    Parameters:
    - image: The original image as a NumPy array.
    - mask: The image mask, with the object of interest in white (255) and the background in black (0).
    - margin: Additional margin in pixels to add around the object (20 pixels by default).

    Returns:
    - The cropped image containing only the object with additional margins.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    x, y = max(0, x - margin), max(0, y - margin)
    w, h = min(image.shape[1] - x, w + 2 * margin), min(image.shape[0] - y, h + 2 * margin)
    return image[y:y+h, x:x+w]

def analyze_mask_color(mask, image, debug=False, debug_folder="debug_images", debug_prefix="debug"):
    """
    Analyzes the color of the object in the mask by comparing the amount of white and red pixels.

    Parameters:
    - mask: The mask of the object.
    - image: The original image.
    - debug: Whether to save debug images.
    - debug_folder: Folder to save debug images.
    - debug_prefix: Prefix for debug image filenames.

    Returns:
    - The predominant color of the object ('white', 'red', or 'undefined').
    """
    if debug and not os.path.exists(debug_folder):
        os.makedirs(debug_folder)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv_image, np.array([0, 0, 150], np.uint8), np.array([180, 45, 255], np.uint8))
    red_mask = cv2.bitwise_or(cv2.inRange(hsv_image, np.array([0, 100, 100], np.uint8), np.array([10, 255, 255], np.uint8)),
                              cv2.inRange(hsv_image, np.array([170, 100, 100], np.uint8), np.array([180, 255, 255], np.uint8)))
    color = 'white' if cv2.countNonZero(white_mask) > cv2.countNonZero(red_mask) else 'red'
    if debug:
        debug_image = cv2.putText(masked_image.copy(), f"Predominantly {color}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(f"{debug_folder}/{debug_prefix}_{np.random.randint(10000)}_{color}.png", debug_image)
    return color

def analyze_frame(frame, model, depth_estimator, output_folder, original_name, debug=False, debug_folder="debug_images"):
    """
    Analyzes a single frame for bowling pins, highlighting red pins and providing special notifications for them.
    Adds centered text to the frame indicating the status of the red pin.

    Parameters:
    - frame: The frame to analyze.
    - model: The loaded YOLO model for object detection.
    - depth_estimator: The loaded depth estimation model.
    - output_folder: The folder where the processed frames will be saved.
    - original_name: The original name of the media file being processed.
    """
    frame = resize_image(frame)
    results = model(frame)
    result = results[0]  # Assuming single image

    final_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    img_depth = depth_estimator.infer(frame)
    if debug:
        # Verificar y crear el directorio de depuración si no existe
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)
        debug_prefix = "depth"
        depth_image_path = (f"{debug_folder}/{debug_prefix}_{original_name}")
        normalized_img = (img_depth - img_depth.min()) / (img_depth.max() - img_depth.min())

        # Escalar los valores a un rango de 0 a 255
        scaled_img = (normalized_img * 255).astype(np.uint8)
        inverted_img = 255 - scaled_img

        colored_img = cv2.applyColorMap(inverted_img, cv2.COLORMAP_JET)  # Puedes cambiar COLORMAP_JET por otro mapa de colores

        # Guardar la imagen resultante
        cv2.imwrite(depth_image_path, colored_img)

    detections = []

    if result.masks is not None:
        for idx, (mask, bbox, cls) in enumerate(zip(result.masks, result.boxes, result.boxes.cls)):
            np_mask = mask.data.cpu().numpy()[0].astype(np.uint8)
            final_mask = cv2.bitwise_or(final_mask, np_mask)
            if cls == 0 or cls == 2:  # Assuming class 0 for people and 2 for car
                depth = analyze_depth(img_depth, np_mask)
                bbox = bbox.data.tolist() # Convert bbox info to list to be serialized
                detections.append({"idx": idx, "depth": float(depth), "class": result.names[int(cls)], "bbox": bbox})
                
    # Save the frame with the added text
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, f"processed_{original_name}")
    result.save(filename=output_path)

    return json.dumps(detections)

def main(media_folder, output_folder="processed_media"):
    """
    Analyzes all media files (images or videos) in a given directory.

    Parameters:
    - media_folder: The directory containing media files to analyze.
    - output_folder: The directory where processed media files will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize the depth estimator and YOLO model
    depth_model_path = 'local::./DepthAnything/checkpoints/depth_anything_metric_depth_outdoor.pt'
    global_settings = {
        'FL': 700.819, 'FY': 771.2, 'FX': 672.2,
        'NYU_DATA': False, 'FINAL_HEIGHT': 480, 'FINAL_WIDTH': 640, 'DATASET': 'nyu',
        'pretrained_resource': depth_model_path
    }
    depth_estimator = DA_estimator('zoedepth', global_settings)
    model = YOLO("yolov8x-seg.pt")

    media_files = glob.glob(os.path.join(media_folder, "*"))
    for media_path in media_files:
        original_name = os.path.basename(media_path)
        if media_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(media_path)
            json_dict = analyze_frame(img, model, depth_estimator, output_folder, original_name, debug=True)
        else:
            cap = cv2.VideoCapture(media_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                json_dict = analyze_frame(frame, model, depth_estimator, output_folder, f"frame_{cap.get(cv2.CAP_PROP_POS_FRAMES):04d}.png", debug=True)
    
    return json_dict

if __name__ == "__main__":
    clear_debug_folder()
    main("input")
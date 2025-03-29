import os
import cv2
import uuid
import torch
import gfpgan
import requests
import tempfile
import warnings
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime
from colorpaws import ColorPaws
from keras.models import load_model
from scipy.ndimage import gaussian_filter
from scripts.layers import AdaIN, AdaptiveAttention
from tensorflow_addons.layers import InstanceNormalization
from moviepy.editor import VideoFileClip, ImageSequenceClip
from scripts.models import FPN, SSH, BboxHead, LandmarkHead, ClassHead
from scripts.utils import norm_crop, estimate_norm, inverse_estimate_norm, transform_landmark_points, get_lm

class FacemakerFM1:
    """Copyright (C) 2025 Ikmal Said. All rights reserved"""
    
    def __init__(self, mode='default'):
        """
        Initialize the FacemakerLegacy class.

        Parameters:
            mode (str): The mode to run the class in. Can be 'default', 'api' or 'webui'.
        """
        # Initialize logging
        self.logger = ColorPaws(self.__class__.__name__, log_on=True)
        
        # Configure environment
        warnings.filterwarnings('ignore')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        # Set device
        if tf.config.list_physical_devices('GPU'):
            self.device = 'GPU'
            gpu_list = tf.config.list_physical_devices('GPU')
            
            if gpu_list:
                for gpu in gpu_list:
                    tf.config.experimental.set_memory_growth(gpu, True)
        else:
            self.device = 'CPU'
        
        self.logger.info(f"Current Torch device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        self.logger.info(f"Current TensorFlow device: {self.device}")
        
        # Initialize models dict
        self.fm1_models = {
            "arcface"   : "models/arcface_fm1.ism",
            "retinaface": "models/retinaface_fm1.ism",
            "facemaker" : "models/facemaker_fm1.ism",
            "resnet50"  : "models/resnet50_fm1.ism",
            "parsenet"  : "models/parsenet_fm1.ism",
            "gfpgan13"  : "models/gfpgan_1.3_fm1.ism",
            "gfpgan14"  : "models/gfpgan_1.4_fm1.ism"
        }
        
        # Load models
        self.__download_models()
        self.__load_models()

        self.logger.info(f"{self.__class__.__name__} is ready!")
        if mode != 'default':
            if mode == 'api':
                self.start_api()
            elif mode == 'webui':
                self.start_webui()

    def start_api(self, host: str = "0.0.0.0", port: int = None, debug: bool = False):
        """
        Start API server with all endpoints.

        Parameters:
        - host (str): Host to run the server on (default: "0.0.0.0")
        - port (int): Port to run the server on (default: None)
        - debug (bool): Enable Flask debug mode (default: False)
        """
        from api import FacemakerWebAPI
        FacemakerWebAPI(self, host=host, port=port, debug=debug)

    def start_webui(self, host: str = None, port: int = None, browser: bool = False, upload_size: str = "100MB",
                    public: bool = False, limit: int = 10, quiet: bool = False):
        """
        Start WebUI with all features.
        
        Parameters:
        - host (str): Server host (default: None)
        - port (int): Server port (default: None) 
        - browser (bool): Launch browser automatically (default: False)
        - upload_size (str): Maximum file size for uploads (default: "10MB")
        - public (bool): Enable public URL mode (default: False)
        - limit (int): Maximum number of concurrent requests (default: 10)
        - quiet (bool): Enable quiet mode (default: False)
        """
        from webui import FacemakerWebUI
        FacemakerWebUI(self, host=host, port=port, browser=browser, upload_size=upload_size,
                       public=public, limit=limit, quiet=quiet)

    def __download_models(self):
        """Downloads model files from Huggingface if they don't exist locally."""
        base_url = "https://huggingface.co/ikmalsaid/facemaker/resolve/main/models/"
        os.makedirs("models", exist_ok=True)
        
        for _, filename in self.fm1_models.items():
            model_path = filename
            model_file = filename.replace("models/", "")
            
            # Skip if file already exists
            if os.path.exists(model_path):
                self.logger.info(f"Model {model_path} already exists! Skipping download...")
                continue
            
            # Download the file
            url = f"{base_url}{model_file}?download=true"        
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            # Show progress bar while downloading
            with open(model_path, 'wb') as f, tqdm(
                desc=f"Downloading {model_file}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    bar.update(size)

    def __load_models(self):
        """Load all required models"""
        with tf.device(self.device):
            self.arcface = load_model(self.fm1_models["arcface"])
            
            self.retinaface = load_model(self.fm1_models["retinaface"], 
                                         custom_objects={"FPN": FPN, "SSH": SSH, "BboxHead": BboxHead, 
                                         "LandmarkHead": LandmarkHead, "ClassHead": ClassHead}
                                         )
            
            self.swapper = load_model(self.fm1_models["facemaker"], 
                                      custom_objects={"AdaIN": AdaIN, "AdaptiveAttention": AdaptiveAttention, 
                                      "InstanceNormalization": InstanceNormalization}
                                      )
            
            self.enhancer = gfpgan.GFPGANer(model_path=self.fm1_models["gfpgan13"], upscale=1)

    def get_taskid(self):
        """
        Generate a unique task ID for request tracking.
        Returns a combination of timestamp and UUID to ensure uniqueness.
        Format: YYYYMMDD_HHMMSS_UUID8
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        uuid_part = str(uuid.uuid4())[:8]
        task_id = f"{timestamp}_{uuid_part}"
        return task_id

    def change_gfpgan(self, model_name):
        """Change GFPGAN model version"""
        gfpgan_models = {
            "GFPGAN 1.3": self.fm1_models["gfpgan13"],
            "GFPGAN 1.4": self.fm1_models["gfpgan14"],
        }
        model_path = gfpgan_models[model_name]
        self.logger.info(f"GFPGAN set to {model_name}")
        
        with tf.device(self.device):
            self.enhancer = gfpgan.GFPGANer(model_path=model_path, upscale=1)

    def detect_faces(self, image):
        """Scans an image for faces, captures them and sorts them in an array.

        Parameters:
            image (str): file path for the image.
        """        
        temp_dir = tempfile.mkdtemp(prefix='fm1_')
        task_id = self.get_taskid()
        base_filename = os.path.splitext(os.path.basename(image))[0]

        # Load the image from file
        image_array = np.array(Image.open(image))
        
        # Initialize lists and counters
        faces_list = []
        faces_index = 0
        
        # Normalize image
        image_array = image_array.astype(np.float32)
        
        if len(image_array.shape) == 2:  # Grayscale image
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[2] == 4:  # RGBA image
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        # Detect faces using Retinaface model
        images = self.retinaface(np.expand_dims(image_array, axis=0)).numpy()
        
        # Convert the detected faces array to numpy array and sort by face area
        scanned_array = np.array(images)
        sorted_indices = np.argsort(scanned_array.sum(axis=1))
        images = scanned_array[sorted_indices]

        # Process each detected face
        for i, image_a in enumerate(images):
            # Get image dimensions and landmarks
            image_h, image_w, _ = image_array.shape
            image_lm = get_lm(image_a, image_w, image_h)
            
            # Align the detected face
            image_aligned = norm_crop(image_array, image_lm, image_size=256)

            # Convert the aligned image from BGR to RGB for saving
            image_aligned = cv2.cvtColor(image_aligned, cv2.COLOR_BGR2RGB)

            # First save with UUID (temporary file)
            temp_file = os.path.join(temp_dir, f"{uuid.uuid4()}.jpg")
            cv2.imwrite(temp_file, image_aligned)
            
            # Create final filename with task_id and face number
            final = os.path.join(temp_dir, f"{task_id}_{base_filename}_face_{i}.jpg")
            os.rename(temp_file, final)
            
            # Append the file path to the list
            faces_list.append(final)
        
        return faces_list, len(faces_list), faces_index

    def swap_frame(self, target_image, source_image, target_index=0, source_index=0, swap_all=False):
        """Swaps faces in the target image with the source face.
        
        Parameters:
            target_image: Path to target image or numpy array
            source_image: Path to source image or numpy array
            target_index: Index of the face to swap in target image
            source_index: Index of the face to use from source image
            swap_all: Whether to swap all faces in the target image
        """
        # Load source and target images, handling both file paths and numpy arrays
        source = np.array(Image.open(source_image)) if isinstance(source_image, str) else source_image
        target = np.array(Image.open(target_image)) if isinstance(target_image, str) else target_image

        # Prepare blend mask with Gaussian smoothing
        blend_mask_base = np.zeros(shape=(256, 256, 1))
        blend_mask_base[80:244, 32:224] = 1
        blend_mask_base = gaussian_filter(blend_mask_base, sigma=7)

        # Process source image
        source_a = self.retinaface(np.expand_dims(source, axis=0)).numpy()

        # Sort detected source faces and select the specified index
        source_array = np.array(source_a)
        sorted_indices = np.argsort(source_array.sum(axis=1))
        source_ax = source_array[sorted_indices]
        source_a = source_ax[source_index]

        # Align the source face
        source_h, source_w, _ = source.shape
        source_lm = get_lm(source_a, source_w, source_h)
        source_aligned = norm_crop(source, source_lm, image_size=256)
        source_z = self.arcface.predict(np.expand_dims(tf.image.resize(source_aligned, [112, 112]) / 255.0, axis=0))

        # Detect faces in the target image
        im = target
        im_h, im_w, _ = im.shape
        im_shape = (im_w, im_h)
        detection_scale = im_w // 640 if im_w > 640 else 1

        faces = self.retinaface(np.expand_dims(cv2.resize(im, (im_w // detection_scale, im_h // detection_scale)), axis=0)).numpy()

        # Sort detected target faces
        faces_list = []
        target_array = np.array(faces)
        sorted_indices = np.argsort(target_array.sum(axis=1))
        faces = target_array[sorted_indices]

        # Check if any faces were detected
        if len(faces) == 0:
            self.logger.warning("No faces detected in target image")
            final = os.path.join(tempfile.mkdtemp(), f"{uuid.uuid4()}.jpg")
            cv2.imwrite(final, cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
            return final
        
        # Ensure target_index is within bounds
        target_index = min(target_index, len(faces) - 1)
        faces_list.append(faces[target_index])

        if swap_all: faces_list = faces
        total_img = im / 255.0

        # Process each detected face for swapping
        for annotation in faces_list:
            lm_align = np.array([[annotation[4] * im_w, annotation[5] * im_h],
                                [annotation[6] * im_w, annotation[7] * im_h],
                                [annotation[8] * im_w, annotation[9] * im_h],
                                [annotation[10] * im_w, annotation[11] * im_h],
                                [annotation[12] * im_w, annotation[13] * im_h]],
                                dtype=np.float32)

            # Align the detected face
            M, _ = estimate_norm(lm_align, 256, "arcface", shrink_factor=1.0)
            im_aligned = (cv2.warpAffine(im, M, (256, 256), borderValue=0.0) - 127.5) / 127.5

            # Perform face swap
            changed_face_cage = self.swapper.predict([np.expand_dims(im_aligned, axis=0), source_z])
            changed_face = changed_face_cage[0] * 0.5 + 0.5

            # Get inverse transformation landmarks
            transformed_lmk = transform_landmark_points(M, lm_align)

            # Warp swapped face back to the original image
            iM, _ = inverse_estimate_norm(lm_align, transformed_lmk, 256, "arcface", shrink_factor=1.0)
            iim_aligned = cv2.warpAffine(changed_face, iM, im_shape, borderValue=0.0)

            # Blend the swapped face with the target image
            blend_mask = cv2.warpAffine(blend_mask_base, iM, im_shape, borderValue=0.0)
            blend_mask = np.expand_dims(blend_mask, axis=-1)
            total_img = (iim_aligned * blend_mask + total_img * (1 - blend_mask))

        # Clip values to valid range and convert to uint8
        total_img = np.clip(total_img, 0, 1)
        total_img *= 255.0
        total_img = total_img.astype('uint8')
        
        # Converts swapped image from BGR to RGB
        total_img = cv2.cvtColor(total_img, cv2.COLOR_BGR2RGB)

        final = os.path.join(tempfile.mkdtemp(), f"{uuid.uuid4()}.jpg")
        cv2.imwrite(final, total_img)
        return final

    def enhance_frame(self, image):
        """Enhances face quality in an image using GFPGAN.
        
        Parameters:
            image: Path to the image or numpy array
        """    
        # Read the image if it's a path
        if isinstance(image, str):
            if not os.path.exists(image):
                raise ValueError(f"Image path does not exist: {image}")
            img = cv2.imread(image)
        else:
            img = image
        
        # Apply face enhancement
        _, _, result = self.enhancer.enhance(img, paste_back=True)
        
        output_path = os.path.join(tempfile.mkdtemp(), f"{uuid.uuid4()}.jpg")
        cv2.imwrite(output_path, result)
        return output_path

    def swap_image(self, target_image, source_image, target_index=0, source_index=0, swap_all=False, face_restore=False, face_restore_model=None):
        """Combines face swapping and enhancement in one function.
        
        Parameters:
            target_image: Path to target image or numpy array
            source_image: Path to source image or numpy array
            target_index: Index of the face to swap in target image
            source_index: Index of the face to use from source image
            face_restore: Whether to enhance face quality
            swap_all: Whether to swap all faces in the target image
            face_restore_model: GFPGAN model to use
        """
        self.logger.info(f"Processing image: {target_image}")
        if face_restore: self.change_gfpgan(face_restore_model)

        # Generate task_id for the final output
        task_id = self.get_taskid()
        base_filename = os.path.splitext(os.path.basename(target_image))[0]

        # First swap the face (using temporary UUID filename)
        swapped_image_path = self.swap_frame(
            target_image=target_image,
            source_image=source_image,
            target_index=target_index,
            source_index=source_index,
            swap_all=swap_all
        )
        
        if face_restore:
            # Enhance using temporary UUID filename
            enhanced_image_path = self.enhance_frame(swapped_image_path)
            os.remove(swapped_image_path)
            
            # Rename the final output with task_id
            final_path = os.path.join(os.path.dirname(enhanced_image_path), 
                                     f"{task_id}_{base_filename}_swapped.jpg")
            os.rename(enhanced_image_path, final_path)
            return final_path
        else:
            # Rename the final output with task_id
            final_path = os.path.join(os.path.dirname(swapped_image_path), 
                                     f"{task_id}_{base_filename}_swapped.jpg")
            os.rename(swapped_image_path, final_path)
            return final_path

    def swap_video(self, source_image, source_index, target_video, face_restore=False, face_restore_model=None):
        """Swaps faces in video frames with a source face and optionally enhances them.
        
        Parameters:
            source_image: Path to source image
            source_index: Index of the face to use from source image
            target_video: Path to target video file
            face_restore: Whether to enhance face quality
            face_restore_model: GFPGAN model to use
        """
        self.logger.info(f"Processing video: {target_video}")
        if face_restore: self.change_gfpgan(face_restore_model)
        
        # Generate task_id for the final output
        task_id = self.get_taskid()
        base_filename = os.path.splitext(os.path.basename(target_video))[0]
        
        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp()
        processed_frames = []
        
        # Load the video
        video = VideoFileClip(target_video)
        
        # Process each frame (using temporary UUID filenames)
        for frame in video.iter_frames():        
            swapped_frame = self.swap_frame(
                target_image=frame,
                source_image=source_image,
                source_index=source_index,
                swap_all=True
            )
            
            if face_restore:
                processed_frame = self.enhance_frame(swapped_frame)
                os.remove(swapped_frame)
            else:
                processed_frame = swapped_frame
            
            processed_frames.append(processed_frame)

        # Create final output with task_id naming
        output_path = os.path.join(tempfile.mkdtemp(), f"{task_id}_{base_filename}_swapped.mp4")
        
        # Combine frames with original audio
        new_clip = ImageSequenceClip(processed_frames, fps=video.fps)
        if video.audio is not None:
            new_clip = new_clip.set_audio(video.audio)
        
        # Write final video
        new_clip.write_videofile(output_path, 
                               codec='libx264', 
                               audio_codec='aac',
                               temp_audiofile=f'{temp_dir}/temp-audio.m4a',
                               remove_temp=True)
        
        # Clean up
        video.close()
        os.rmdir(temp_dir)
        
        return output_path
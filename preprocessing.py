import os
import torch
import numpy as np
import cv2
import librosa
import librosa.display
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import warnings
from fastai.vision.all import DataBlock, ImageBlock, BBoxLblBlock, BBoxLbl, MaskBlock, RandomSplitter, Resize, Normalize, PILImage
import torchvision.models.detection as detection
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, FastRCNNPredictor
from moviepy.editor import VideoFileClip
from fastai.vision.all import Learner, accuracy

# Suppress warnings
warnings.filterwarnings("ignore")

# Define paths
VIDEO_DIR = Path('D:/Minor project/dataset')
TRANSCRIPT_DIR = Path('D:/Minor project/dataset_labels')

def create_segmentation_mask(image_path, objects):
    """
    Create a segmentation mask for each object in the image.
    
    :param image_path: Path to the image.
    :param objects: List of objects with their bounding boxes.
    :return: Segmentation mask.
    """
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    mask = Image.new("L", (width, height), 0)
    
    draw = ImageDraw.Draw(mask)
    
    for obj in objects:
        bbox = obj['bbox']
        draw.rectangle(bbox, outline=255, fill=255)
    
    return mask

def extract_audio_features(video_file):
    try:
        video = VideoFileClip(str(video_file))
        audio = video.audio
        if audio is None:
            print(f"No audio found in {video_file}")
            return torch.zeros(76, dtype=torch.float32), None
        
        # Save audio to a temporary file
        temp_audio_file = "temp_audio.wav"
        audio.write_audiofile(temp_audio_file, logger=None)
        
        # Load audio file with librosa
        y, sr = librosa.load(temp_audio_file, duration=None)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Compute statistics over time
        features = np.concatenate([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(chroma, axis=1), np.std(chroma, axis=1),
            np.mean(spectral_contrast, axis=1), np.std(spectral_contrast, axis=1)
        ])
        
        # Remove temporary audio file
        os.remove(temp_audio_file)
        
        return torch.tensor(features, dtype=torch.float32), mfcc
    except Exception as e:
        print(f"Error extracting audio features: {str(e)}")
        return torch.zeros(76, dtype=torch.float32), None


def extract_frames(video_path, target_size=(224, 224), sample_interval=30):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, target_size)
        frames.append(frame)
    
    cap.release()
    sampled_frames = [frames[i] for i in range(0, len(frames), sample_interval)]
    return sampled_frames

def frames_to_image_paths(frames, save_path='frames/'):
    os.makedirs(save_path, exist_ok=True)
    image_paths = []
    for i, frame in enumerate(frames):
        path = os.path.join(save_path, f'frame_{i}.jpg')
        PILImage.create(frame).save(path)
        image_paths.append(path)
    return image_paths

def create_segmentation_datablock(image_paths, bboxes, masks, labels, size=(224, 224), batch_size=32):
    datablock = DataBlock(
        blocks=(ImageBlock, BBoxLblBlock, MaskBlock),
        get_items=lambda x: image_paths,
        get_y=lambda x: [
            (BBoxLbl([bbox], [label]), mask) 
            for bbox, mask, label in zip(bboxes, masks, labels)
        ],
        splitter=RandomSplitter(),
        item_tfms=Resize(size),
        batch_tfms=[Normalize.from_stats(*PILImage.create(image_paths[0]).stats)]
    )
    return datablock.dataloaders(image_paths, bs=batch_size)

def load_mask_rcnn_model(num_classes):
    model = detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_mask_features = model.roi_heads.mask_predictor.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask_features, num_classes)
    
    return model

def create_learner(dataloaders, model):
    def mask_rcnn_loss(pred, targets):
        losses = model.roi_heads.losses(pred, targets)
        return losses['loss_box_reg'] + losses['loss_objectness'] + losses['loss_rpn_box_reg'] + losses['loss_rpn_objectness'] + losses['loss_mask']
    
    learn = Learner(dataloaders, model, loss_func=mask_rcnn_loss, metrics=[accuracy])
    return learn
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def extract_text_features(transcript):
    tfidf_vectorizer = TfidfVectorizer(max_features=18)
    features = tfidf_vectorizer.fit_transform([transcript]).toarray()[0]
    return torch.tensor(features, dtype=torch.float32)

def process_video(video_file, transcript):
    frames = extract_frames(video_file)
    image_paths = frames_to_image_paths(frames)
    
    # Example bounding boxes and masks; replace with actual data
    bboxes = [
        [[50, 50, 100, 100]],  # Replace with real bounding boxes
        [[30, 30, 80, 80]]
    ]
    masks = [create_segmentation_mask(p, [{'bbox': b} for b in bbox]) for p, bbox in zip(image_paths, bboxes)]
    labels = [0] * len(image_paths)  # Example labels
    
    # Create DataLoaders for object detection and segmentation
    dataloaders = create_segmentation_datablock(image_paths, bboxes, masks, labels, size=(224, 224), batch_size=32)
    
    # Load and configure Mask R-CNN model
    num_classes = len(set(labels)) + 1
    model = load_mask_rcnn_model(num_classes)
    
    # Define the Learner
    def mask_rcnn_loss(pred, targets):
        return detection.maskrcnn_loss(pred, targets)
    
    learn = Learner(dataloaders, model, loss_func=mask_rcnn_loss, metrics=[])

    # Predict
    pred = [learn.predict(PILImage.create(img_path)) for img_path in image_paths]
    
    # Extract audio and text features
    audio_features, mfcc = extract_audio_features(video_file)
    text_features = extract_text_features(transcript)
    
    print(f"Audio features shape: {audio_features.shape}")
    print(f"Text features shape: {text_features.shape}")
    
    # Combine features
    combined_features = torch.cat([audio_features, text_features])
    
    return audio_features, combined_features, pred, mfcc, frames[0]


def display_image_and_audio(image, mfcc, video_name):
    if image is None or mfcc is None:
        print(f"Cannot display visualization for {video_name}: missing image or audio data")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("First Frame")
    plt.axis('off')
    
    # Display MFCC
    plt.subplot(1, 2, 2)
    librosa.display.specshow(mfcc, x_axis='time', sr=22050)
    plt.colorbar()
    plt.title("MFCC")
    
    plt.suptitle(f"Visualization for {video_name}")
    plt.tight_layout()
    plt.show()


def main():
    all_audio_features = []
    all_combined_features = []
    all_predictions = []
    
    for video_file in VIDEO_DIR.glob('*.mp4'):
        print(f"Processing {video_file.name}")
        
        # Get corresponding transcript file
        transcript_file = TRANSCRIPT_DIR / (video_file.stem + '.txt')
        
        if not transcript_file.exists():
            print(f"Transcript file not found for {video_file.name}")
            continue
        
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript = f.read()
        
        audio_features, combined_features, predictions, mfcc, first_frame = process_video(video_file, transcript)
        
        all_audio_features.append(audio_features)
        all_combined_features.append(combined_features)
        all_predictions.append(predictions)
        
        print(f"Combined features shape: {combined_features.shape}")
        print("------------------------")
        
        # Display image and audio visualization
        display_image_and_audio(first_frame, mfcc, video_file.name)
    
    # Stack all features into tensors
    all_audio_features_tensor = torch.stack(all_audio_features)
    all_combined_features_tensor = torch.stack(all_combined_features)
    
    # Save features
    torch.save(all_audio_features_tensor, 'all_audio_features.pt')
    torch.save(all_combined_features_tensor, 'all_combined_features.pt')
    
    print(f"All audio features saved. Shape: {all_audio_features_tensor.shape}")
    print(f"All combined features saved. Shape: {all_combined_features_tensor.shape}")

if __name__ == "__main__":
    main()

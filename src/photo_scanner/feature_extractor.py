"""
Feature extraction using pretrained CNN models.
Extracts visual embeddings from photos for similarity comparison and classification.
"""

import torch
from torchvision import transforms
from PIL import Image

# Register HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass
import timm
from pathlib import Path
from typing import List, Union, Optional
import numpy as np
from tqdm import tqdm


class FeatureExtractor:
    """Extract visual features from images using pretrained models."""
    
    def __init__(self, model_name: str = 'efficientnet_b0', device: str = None):
        """
        Initialize feature extractor.
        
        Args:
            model_name: Name of the pretrained model from timm
            device: 'cuda', 'mps', or 'cpu' (auto-detected if None)
        """
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'  # Apple Silicon
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load pretrained model
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get model's expected input size
        self.input_size = self.model.default_cfg.get('input_size', (3, 224, 224))[-1]
        
        # Define preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.input_size, self.input_size).to(self.device)
            self.feature_dim = self.model(dummy).shape[1]
        
        print(f"Feature dimension: {self.feature_dim}")
    
    def load_image(self, image_path: Union[str, Path]) -> Optional[torch.Tensor]:
        """Load and preprocess an image."""
        try:
            img = Image.open(image_path).convert('RGB')
            return self.transform(img)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    @torch.no_grad()
    def extract_single(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Extract features from a single image."""
        tensor = self.load_image(image_path)
        if tensor is None:
            return None
        
        tensor = tensor.unsqueeze(0).to(self.device)
        features = self.model(tensor)
        return features.cpu().numpy().flatten()
    
    @torch.no_grad()
    def extract_batch(self, image_paths: List[Union[str, Path]], 
                      batch_size: int = 32,
                      show_progress: bool = True) -> dict:
        """
        Extract features from multiple images.
        
        Returns:
            Dictionary mapping image paths to feature vectors
        """
        results = {}
        
        # Process in batches
        iterator = range(0, len(image_paths), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting features", 
                          total=len(image_paths) // batch_size + 1)
        
        for i in iterator:
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            valid_paths = []
            
            for path in batch_paths:
                tensor = self.load_image(path)
                if tensor is not None:
                    batch_tensors.append(tensor)
                    valid_paths.append(path)
            
            if batch_tensors:
                batch = torch.stack(batch_tensors).to(self.device)
                features = self.model(batch).cpu().numpy()
                
                for path, feat in zip(valid_paths, features):
                    results[str(path)] = feat
        
        return results


class AestheticScorer:
    """
    Score images based on learned aesthetic preferences.
    Uses the extracted features to predict how much the user would like a photo.
    """
    
    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor
        self.model = None
        self.scaler = None
        self.is_one_class = True
    
    def train(self, good_features: np.ndarray, bad_features: np.ndarray = None):
        """
        Train the aesthetic scorer.
        
        If bad_features is provided, trains a binary classifier (good vs bad).
        Otherwise, uses one-class learning to identify what "good" looks like.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import OneClassSVM
        from sklearn.ensemble import RandomForestClassifier
        
        self.scaler = StandardScaler()
        good_scaled = self.scaler.fit_transform(good_features)
        
        if bad_features is None or len(bad_features) < 10:
            # One-class classification - learn what "good" looks like
            print(f"Training one-class model on {len(good_features)} curated photos...")
            self.model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
            self.model.fit(good_scaled)
            self.is_one_class = True
        else:
            # Binary classification - learn to distinguish good from bad
            print(f"Training binary classifier: {len(good_features)} good, {len(bad_features)} bad...")
            bad_scaled = self.scaler.transform(bad_features)
            
            X = np.vstack([good_scaled, bad_scaled])
            y = np.array([1] * len(good_scaled) + [0] * len(bad_scaled))
            
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            self.model.fit(X, y)
            self.is_one_class = False
        
        print("Training complete!")
    
    def score(self, features: np.ndarray) -> np.ndarray:
        """
        Score features. Higher = more similar to your curated photos.
        
        Returns scores between 0 and 1.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        scaled = self.scaler.transform(features.reshape(-1, self.feature_extractor.feature_dim))
        
        if self.is_one_class:
            # One-class SVM returns distance from boundary
            # Positive = inside (similar to training), negative = outside (different)
            raw_scores = self.model.decision_function(scaled)
            # Normalize to 0-1 range using sigmoid
            scores = 1 / (1 + np.exp(-raw_scores))
        else:
            # Binary classifier - probability of being "good"
            scores = self.model.predict_proba(scaled)[:, 1]
        
        return scores.flatten()
    
    def save(self, path: Union[str, Path]):
        """Save the trained model."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'is_one_class': self.is_one_class,
            }, f)
        print(f"Model saved to {path}")
    
    def load(self, path: Union[str, Path]):
        """Load a trained model."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_one_class = data.get('is_one_class', True)  # Default to one-class for old models
        print(f"Model loaded from {path}")

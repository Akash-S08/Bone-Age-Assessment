import pytest
import torch
import numpy as np
import tempfile
import os
import json
from PIL import Image

import sys
sys.path.append('../backend')

from model import BoneAgeNet, create_model
from dataset import BoneAgeDataset, generate_sample_data
from utils import preprocess_image, create_attention_heatmap

class TestBoneAgeSystem:
    
    @pytest.fixture
    def sample_config(self):
        return {
            "model": {
                "backbone": "resnet34",
                "num_classes": 1,
                "attention_type": "cbam",
                "pretrained": False
            },
            "data": {
                "image_size": 224,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        }
    
    @pytest.fixture
    def sample_image(self):
        # Create a sample RGB image
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        return image
    
    def test_model_creation(self, sample_config):
        """Test model creation and forward pass"""
        model = create_model(sample_config)
        assert isinstance(model, BoneAgeNet)
        
        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        
        assert 'age_pred' in output
        assert 'attention_map' in output
        assert output['age_pred'].shape == (1, 1)
        assert len(output['attention_map'].shape) == 4  # Batch, channels, H, W
    
    def test_image_preprocessing(self, sample_image):
        """Test image preprocessing"""
        processed = preprocess_image(sample_image, target_size=224)
        
        assert isinstance(processed, torch.Tensor)
        assert processed.shape == (1, 3, 224, 224)
        assert processed.dtype == torch.float32
    
    def test_attention_heatmap_creation(self, sample_image):
        """Test attention heatmap creation"""
        # Create dummy attention map
        attention_map = torch.randn(1, 1, 7, 7)
        
        overlay, overlay_b64 = create_attention_heatmap(attention_map, sample_image)
        
        assert isinstance(overlay, np.ndarray)
        assert isinstance(overlay_b64, str)
        assert overlay.shape == sample_image.shape
    
    def test_dataset_creation(self):
        """Test dataset creation with synthetic data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate sample data
            generate_sample_data(temp_dir, num_samples=10)
            
            # Create dataset
            dataset = BoneAgeDataset(
                data_dir=os.path.join(temp_dir, 'train'),
                csv_file=os.path.join(temp_dir, 'train_labels.csv'),
                image_size=224
            )
            
            assert len(dataset) > 0
            
            # Test data loading
            image, age = dataset[0]
            assert isinstance(image, np.ndarray)
            assert isinstance(age, (int, float))

def run_tests():
    """Run all tests"""
    pytest.main([__file__, "-v"])

if __name__ == "__main__":
    run_tests()
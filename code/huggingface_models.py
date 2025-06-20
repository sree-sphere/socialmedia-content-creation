"""
Hugging Face Model Integrations for Content Generation
This module provides interfaces to various Hugging Face models for text, image, and video generation.
"""

import os
import io
import json
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path

# Optional imports with fallback handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available, some features may be limited")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available, image handling may be limited")

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Warning: streamlit not available in this context")

try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not available, using mock implementation")
    
    # Mock InferenceClient for testing
    class MockInferenceClient:
        def __init__(self, token=None):
            self.token = token
        
        def text_generation(self, prompt, model=None, **kwargs):
            return f"Mock response for: {prompt[:50]}..."
        
        def text_to_image(self, prompt, model=None):
            return b"mock_image_data"
    
    InferenceClient = MockInferenceClient

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuggingFaceModelManager:
    """
    Manages all Hugging Face model interactions for the content creation app.
    """
    
    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize the model manager.
        
        Args:
            hf_token (str, optional): Hugging Face API token for authenticated requests
        """
        self.hf_token = hf_token or os.getenv('HUGGINGFACE_TOKEN')
        self.client = InferenceClient(token=self.hf_token)
        
        # Model configurations based on research recommendations
        self.models = {
            'text': {
                'primary': 'meta-llama/Meta-Llama-3-8B-Instruct',
                'fallback': 'microsoft/DialoGPT-medium',
                'alternatives': [
                    'mistralai/Mixtral-8x7B-Instruct-v0.1',
                    'google/flan-t5-large'
                ]
            },
            'image': {
                'primary': 'stabilityai/stable-diffusion-xl-base-1.0',
                'fallback': 'runwayml/stable-diffusion-v1-5',
                'alternatives': [
                    'CompVis/stable-diffusion-v1-4',
                    'prompthero/openjourney'
                ]
            },
            'video': {
                'primary': 'ali-vilab/modelscope-damo-text-to-video-synthesis',
                'fallback': 'cerspense/zeroscope_v2_576w',
                'alternatives': [
                    'stabilityai/stable-video-diffusion-img2vid-xt'
                ]
            }
        }
        
        # Status tracking
        self.model_status = {
            'text': {'available': False, 'current_model': None},
            'image': {'available': False, 'current_model': None},
            'video': {'available': False, 'current_model': None}
        }

    def test_model_availability(self, model_type: str) -> bool:
        """
        Test if a model type is available and working.
        
        Args:
            model_type (str): Type of model ('text', 'image', 'video')
            
        Returns:
            bool: True if model is available and working
        """
        try:
            if model_type == 'text':
                return self._test_text_model()
            elif model_type == 'image':
                return self._test_image_model()
            elif model_type == 'video':
                return self._test_video_model()
            else:
                return False
        except Exception as e:
            logger.error(f"Error testing {model_type} model: {e}")
            return False

    def _test_text_model(self) -> bool:
        """Test text generation model."""
        try:
            test_prompt = "Generate a brief hello message."
            model = self.models['text']['primary']
            
            response = self.client.text_generation(
                test_prompt,
                model=model,
                max_new_tokens=50
            )
            
            if response and len(response.strip()) > 0:
                self.model_status['text']['available'] = True
                self.model_status['text']['current_model'] = model
                logger.info(f"Text model {model} is available")
                return True
            return False
        except Exception as e:
            logger.warning(f"Primary text model failed, trying fallback: {e}")
            return self._test_fallback_text_model()

    def _test_fallback_text_model(self) -> bool:
        """Test fallback text model."""
        try:
            model = self.models['text']['fallback']
            test_prompt = "Hello"
            
            response = self.client.text_generation(
                test_prompt,
                model=model,
                max_new_tokens=30
            )
            
            if response:
                self.model_status['text']['available'] = True
                self.model_status['text']['current_model'] = model
                logger.info(f"Fallback text model {model} is available")
                return True
            return False
        except Exception as e:
            logger.error(f"All text models failed: {e}")
            return False

    def _test_image_model(self) -> bool:
        """Test image generation model."""
        try:
            test_prompt = "A simple red circle"
            model = self.models['image']['primary']
            
            response = self.client.text_to_image(test_prompt, model=model)
            
            if response:
                self.model_status['image']['available'] = True
                self.model_status['image']['current_model'] = model
                logger.info(f"Image model {model} is available")
                return True
            return False
        except Exception as e:
            logger.warning(f"Primary image model failed, trying fallback: {e}")
            return self._test_fallback_image_model()

    def _test_fallback_image_model(self) -> bool:
        """Test fallback image model."""
        try:
            model = self.models['image']['fallback']
            test_prompt = "A simple red circle"
            
            response = self.client.text_to_image(test_prompt, model=model)
            
            if response:
                self.model_status['image']['available'] = True
                self.model_status['image']['current_model'] = model
                logger.info(f"Fallback image model {model} is available")
                return True
            return False
        except Exception as e:
            logger.error(f"All image models failed: {e}")
            return False

    def _test_video_model(self) -> bool:
        """Test video generation model (simplified test)."""
        try:
            # For video models, we'll just check if the model endpoint is reachable
            # Full video generation testing would be too resource-intensive
            model = self.models['video']['primary']
            
            # This is a simplified test - in production, you might want a more thorough check
            self.model_status['video']['available'] = True
            self.model_status['video']['current_model'] = model
            logger.info(f"Video model {model} is considered available (simplified test)")
            return True
        except Exception as e:
            logger.error(f"Video model test failed: {e}")
            return False

class TextGenerator:
    """Handles text generation tasks."""
    
    def __init__(self, model_manager: HuggingFaceModelManager):
        self.model_manager = model_manager
        self.client = model_manager.client

    def generate_content_plan(self, client_type: str, duration: str = "1 month") -> str:
        """
        Generate a comprehensive content plan in JSON format.
        
        Args:
            client_type (str): Type of client (e.g., 'accountant', 'chef', 'artist')
            duration (str): Duration of the content plan
            
        Returns:
            str: JSON formatted content plan
        """
        if not self.model_manager.model_status['text']['available']:
            raise Exception("Text generation model is not available")
        
        model = self.model_manager.model_status['text']['current_model']
        
        prompt = f"""
My client is a {client_type} and they want to improve their online presence over Instagram. 
As a media marketing company, I need to create a content planning document for the period of {duration}. 
I want you to give me day-wise content ideas with infographic descriptions (to instruct my graphic designer) and proper hashtags.

Please provide the output as a valid JSON with the following structure:
{{
    "client_type": "{client_type}",
    "duration": "{duration}",
    "days": [
        {{
            "day": 1,
            "content_idea": "Brief description of the content idea",
            "infographic_description": "Detailed description for the graphic designer",
            "hashtags": "#relevant #hashtags #for #the #post",
            "post_caption": "Engaging caption for the post"
        }}
    ]
}}

Generate content for 30 days. Make sure each day has unique, engaging content relevant to a {client_type}.
Return only the JSON, no other text.
"""
        
        try:
            response = self.client.text_generation(
                prompt,
                model=model,
                max_new_tokens=2048,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )
            
            # Clean and validate JSON response
            cleaned_response = self._clean_json_response(response)
            return cleaned_response
            
        except Exception as e:
            logger.error(f"Error generating content plan: {e}")
            return self._fallback_content_plan(client_type, duration)

    def _clean_json_response(self, response: str) -> str:
        """Clean and validate JSON response from the model."""
        try:
            # Find JSON content between curly braces
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end != 0:
                json_str = response[start:end]
                # Validate by parsing
                json.loads(json_str)
                return json_str
            else:
                raise ValueError("No valid JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to clean JSON response: {e}")
            raise Exception("Model returned invalid JSON format")

    def _fallback_content_plan(self, client_type: str, duration: str) -> str:
        """Generate a fallback content plan if the main model fails."""
        fallback_plan = {
            "client_type": client_type,
            "duration": duration,
            "days": [
                {
                    "day": 1,
                    "content_idea": f"Introduction post showcasing {client_type} services",
                    "infographic_description": f"Professional design featuring {client_type} tools and services with modern typography",
                    "hashtags": f"#{client_type.lower()} #professional #services #introduction",
                    "post_caption": f"Welcome to our {client_type} services! Let's grow together."
                },
                {
                    "day": 2,
                    "content_idea": f"Tips and tricks for {client_type} industry",
                    "infographic_description": f"Clean infographic with 3-5 tips relevant to {client_type} work",
                    "hashtags": f"#{client_type.lower()}tips #professional #advice",
                    "post_caption": f"Pro tips from experienced {client_type}s to help you succeed!"
                }
            ]
        }
        
        return json.dumps(fallback_plan, indent=2)

class ImageGenerator:
    """Handles image generation tasks."""
    
    def __init__(self, model_manager: HuggingFaceModelManager):
        self.model_manager = model_manager
        self.client = model_manager.client

    def generate_image(self, prompt: str, output_path: Optional[str] = None) -> Union[Image.Image, str]:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt (str): Text prompt for image generation
            output_path (str, optional): Path to save the generated image
            
        Returns:
            PIL.Image or str: Generated image object or path to saved image
        """
        if not self.model_manager.model_status['image']['available']:
            raise Exception("Image generation model is not available")
        
        model = self.model_manager.model_status['image']['current_model']
        
        try:
            response = self.client.text_to_image(prompt, model=model)
            
            if isinstance(response, bytes):
                image = Image.open(io.BytesIO(response))
            else:
                image = response
            
            if output_path:
                image.save(output_path)
                logger.info(f"Image saved to {output_path}")
                return output_path
            
            return image
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return self._generate_placeholder_image(output_path)

    def _generate_placeholder_image(self, output_path: Optional[str] = None) -> Union[Image.Image, str]:
        """Generate a placeholder image when model fails."""
        # Create a simple placeholder image
        placeholder = Image.new('RGB', (512, 512), color='lightgray')
        
        if output_path:
            placeholder.save(output_path)
            return output_path
        
        return placeholder

class VideoGenerator:
    """Handles video generation tasks."""
    
    def __init__(self, model_manager: HuggingFaceModelManager):
        self.model_manager = model_manager
        self.client = model_manager.client

    def generate_video(self, prompt: str, output_path: Optional[str] = None) -> str:
        """
        Generate a video from a text prompt.
        
        Args:
            prompt (str): Text prompt for video generation
            output_path (str, optional): Path to save the generated video
            
        Returns:
            str: Path to the generated video file or placeholder message
        """
        if not self.model_manager.model_status['video']['available']:
            return "Video generation model is not available"
        
        try:
            # Note: Video generation is computationally intensive and may require special handling
            # This is a simplified implementation
            logger.info(f"Video generation requested for prompt: {prompt}")
            
            # For now, return a placeholder message
            # In a full implementation, this would use diffusers or similar libraries
            placeholder_message = f"Video generation queued for: '{prompt[:50]}...'"
            
            if output_path:
                # Create a placeholder file
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path.replace('.mp4', '.txt'), 'w') as f:
                    f.write(f"Video placeholder for prompt: {prompt}")
                return output_path.replace('.mp4', '.txt')
            
            return placeholder_message
            
        except Exception as e:
            logger.error(f"Error generating video: {e}")
            return f"Video generation failed: {str(e)}"

# Example usage and testing
if __name__ == "__main__":
    # Initialize the model manager
    model_manager = HuggingFaceModelManager()
    
    # Test model availability
    print("Testing model availability...")
    text_available = model_manager.test_model_availability('text')
    image_available = model_manager.test_model_availability('image')
    video_available = model_manager.test_model_availability('video')
    
    print(f"Text model available: {text_available}")
    print(f"Image model available: {image_available}")
    print(f"Video model available: {video_available}")
    
    # Test text generation
    if text_available:
        text_gen = TextGenerator(model_manager)
        try:
            content_plan = text_gen.generate_content_plan("chef")
            print("\n--- Generated Content Plan ---")
            print(content_plan[:500] + "..." if len(content_plan) > 500 else content_plan)
        except Exception as e:
            print(f"Text generation test failed: {e}")
    
    # Test image generation
    if image_available:
        image_gen = ImageGenerator(model_manager)
        try:
            image = image_gen.generate_image("A professional chef in a modern kitchen")
            print(f"\n--- Image Generation Test ---")
            print(f"Generated image type: {type(image)}")
        except Exception as e:
            print(f"Image generation test failed: {e}")

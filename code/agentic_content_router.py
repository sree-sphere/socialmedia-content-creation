"""
Agentic AI Content Router for Context Creation App
This module implements the core routing logic for directing content to appropriate generative models.
"""

import json
import re
from typing import Dict, List, Tuple, Any
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Enum for different content types that can be generated."""
    TEXT = "text"
    IMAGE = "image" 
    VIDEO = "video"
    INFOGRAPHIC = "infographic"
    SOCIAL_POST = "social_post"

class ContentRouter:
    """
    Agentic AI router that intelligently routes content to appropriate generative models.
    Uses pattern matching and keyword analysis to determine content types.
    """
    
    def __init__(self):
        self.image_keywords = [
            'infographic', 'graphic', 'image', 'visual', 'picture', 'photo',
            'illustration', 'diagram', 'chart', 'poster', 'banner', 'logo',
            'design', 'artwork', 'thumbnail', 'icon'
        ]
        
        self.video_keywords = [
            'video', 'reel', 'story', 'clip', 'animation', 'motion',
            'film', 'movie', 'tutorial', 'demo', 'commercial', 'ad',
            'timelapse', 'slideshow', 'presentation'
        ]
        
        self.social_keywords = [
            'post', 'caption', 'hashtag', 'tweet', 'story', 'reel',
            'instagram', 'facebook', 'twitter', 'linkedin', 'tiktok'
        ]

    def analyze_content_plan(self, content_plan: str) -> Dict[str, List[Dict]]:
        """
        Analyzes the JSON content plan and routes content to appropriate models.
        
        Args:
            content_plan (str): JSON string containing the content plan
            
        Returns:
            Dict containing routed content for each model type
        """
        try:
            data = json.loads(content_plan)
            routed_content = {
                'text_generation': [],
                'image_generation': [],
                'video_generation': []
            }
            
            # Handle different JSON structures
            if isinstance(data, dict):
                if 'days' in data or 'content_plan' in data:
                    # Weekly/monthly plan structure
                    days_data = data.get('days', data.get('content_plan', []))
                    for day_info in days_data:
                        self._route_day_content(day_info, routed_content)
                else:
                    # Single day structure
                    self._route_day_content(data, routed_content)
            elif isinstance(data, list):
                # List of days
                for day_info in data:
                    self._route_day_content(day_info, routed_content)
            
            logger.info(f"Content routing completed. "
                       f"Text: {len(routed_content['text_generation'])}, "
                       f"Image: {len(routed_content['image_generation'])}, "
                       f"Video: {len(routed_content['video_generation'])}")
            
            return routed_content
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in content plan: {e}")
            return self._empty_routing()
        except Exception as e:
            logger.error(f"Error analyzing content plan: {e}")
            return self._empty_routing()

    def _route_day_content(self, day_data: Dict, routed_content: Dict):
        """Routes content for a single day's data."""
        day = day_data.get('day', 'Unknown')
        
        # Route each field in the day data
        for key, value in day_data.items():
            if key == 'day':
                continue
                
            content_type = self._classify_content(key, value)
            content_item = {
                'day': day,
                'type': key,
                'content': value,
                'original_key': key
            }
            
            if content_type == ContentType.IMAGE:
                routed_content['image_generation'].append(content_item)
            elif content_type == ContentType.VIDEO:
                routed_content['video_generation'].append(content_item)
            else:
                routed_content['text_generation'].append(content_item)

    def _classify_content(self, key: str, value: str) -> ContentType:
        """
        Classifies content based on key and value analysis.
        
        Args:
            key (str): The key from the JSON (e.g., 'infographic_description')
            value (str): The content value
            
        Returns:
            ContentType: The classified content type
        """
        # Convert to lowercase for analysis
        key_lower = key.lower()
        value_lower = value.lower() if isinstance(value, str) else str(value).lower()
        
        # Check for image content
        if any(keyword in key_lower for keyword in self.image_keywords):
            return ContentType.IMAGE
        if any(keyword in value_lower for keyword in self.image_keywords):
            return ContentType.IMAGE
            
        # Check for video content
        if any(keyword in key_lower for keyword in self.video_keywords):
            return ContentType.VIDEO
        if any(keyword in value_lower for keyword in self.video_keywords):
            return ContentType.VIDEO
            
        # Check for social media content (treated as text with special formatting)
        if any(keyword in key_lower for keyword in self.social_keywords):
            return ContentType.SOCIAL_POST
            
        # Default to text
        return ContentType.TEXT

    def _empty_routing(self) -> Dict[str, List]:
        """Returns empty routing structure."""
        return {
            'text_generation': [],
            'image_generation': [],
            'video_generation': []
        }

    def get_routing_summary(self, routed_content: Dict) -> str:
        """
        Generates a summary of the routing results.
        
        Args:
            routed_content (Dict): The routed content dictionary
            
        Returns:
            str: Summary of routing results
        """
        text_count = len(routed_content.get('text_generation', []))
        image_count = len(routed_content.get('image_generation', []))
        video_count = len(routed_content.get('video_generation', []))
        
        summary = f"""
Content Routing Summary:
- Text Generation Tasks: {text_count}
- Image Generation Tasks: {image_count}
- Video Generation Tasks: {video_count}
- Total Tasks: {text_count + image_count + video_count}
        """
        
        return summary.strip()

class SmartPromptBuilder:
    """
    Builds optimized prompts for different generative models based on content type and context.
    """
    
    def __init__(self):
        self.image_prompt_templates = {
            'infographic': "Create a professional infographic for {client_type}: {content}. Style: clean, modern, informative design with clear visual hierarchy.",
            'social_media': "Design a social media graphic for {client_type}: {content}. Style: eye-catching, vibrant, optimized for social platforms.",
            'general': "Create a visual representation for {client_type}: {content}. Style: professional, high-quality, relevant to the business."
        }
        
        self.video_prompt_templates = {
            'promotional': "Create a promotional video for {client_type}: {content}. Style: engaging, professional, with smooth transitions.",
            'tutorial': "Create an instructional video for {client_type}: {content}. Style: clear, educational, step-by-step format.",
            'general': "Create a video content for {client_type}: {content}. Style: professional, engaging, appropriate for business use."
        }

    def build_image_prompt(self, content_item: Dict, client_type: str) -> str:
        """Builds optimized prompt for image generation."""
        content = content_item.get('content', '')
        content_type = content_item.get('type', '').lower()
        
        # Determine template based on content type
        if 'infographic' in content_type:
            template = self.image_prompt_templates['infographic']
        elif any(keyword in content_type for keyword in ['post', 'social', 'caption']):
            template = self.image_prompt_templates['social_media']
        else:
            template = self.image_prompt_templates['general']
            
        return template.format(client_type=client_type, content=content)

    def build_video_prompt(self, content_item: Dict, client_type: str) -> str:
        """Builds optimized prompt for video generation."""
        content = content_item.get('content', '')
        content_type = content_item.get('type', '').lower()
        
        # Determine template based on content type
        if any(keyword in content_type for keyword in ['tutorial', 'how-to', 'guide']):
            template = self.video_prompt_templates['tutorial']
        elif any(keyword in content_type for keyword in ['promo', 'ad', 'commercial']):
            template = self.video_prompt_templates['promotional']
        else:
            template = self.video_prompt_templates['general']
            
        return template.format(client_type=client_type, content=content)

# Example usage and testing
if __name__ == "__main__":
    # Test the router
    router = ContentRouter()
    prompt_builder = SmartPromptBuilder()
    
    # Sample content plan
    sample_plan = '''
    {
        "days": [
            {
                "day": 1,
                "content_idea": "Share a tip about time management for busy professionals",
                "infographic_description": "A visual guide showing 5 time management techniques with icons and brief explanations",
                "hashtags": "#productivity #timemanagement #businesstips",
                "video_script": "Create a 30-second video demonstrating the Pomodoro technique"
            },
            {
                "day": 2,
                "content_idea": "Behind-the-scenes look at daily operations",
                "photo_description": "Professional photo of the workspace with good lighting",
                "caption": "A day in the life of our team #behindthescenes #teamwork"
            }
        ]
    }
    '''
    
    # Route the content
    routed = router.analyze_content_plan(sample_plan)
    
    # Print routing summary
    print(router.get_routing_summary(routed))
    
    # Test prompt building
    print("\n--- Sample Prompts ---")
    if routed['image_generation']:
        sample_image = routed['image_generation'][0]
        prompt = prompt_builder.build_image_prompt(sample_image, "marketing consultant")
        print(f"Image Prompt: {prompt}")
        
    if routed['video_generation']:
        sample_video = routed['video_generation'][0]
        prompt = prompt_builder.build_video_prompt(sample_video, "marketing consultant")
        print(f"Video Prompt: {prompt}")

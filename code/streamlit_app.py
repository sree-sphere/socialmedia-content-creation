"""
Context Creation App with Agentic AI
Main Streamlit application for generating content plans and multimedia content.
"""

import streamlit as st
import json
import os
import tempfile
from pathlib import Path
import time
from typing import Dict, List, Optional
import traceback

# Import our custom modules
from agentic_content_router import ContentRouter, SmartPromptBuilder
from huggingface_models import HuggingFaceModelManager, TextGenerator, ImageGenerator, VideoGenerator

# Configure Streamlit page
st.set_page_config(
    page_title="Context Creation App",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}

.section-header {
    font-size: 1.5rem;
    color: #2e7bcf;
    margin-top: 2rem;
    margin-bottom: 1rem;
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 0.5rem;
}

.status-success {
    color: #28a745;
    font-weight: bold;
}

.status-error {
    color: #dc3545;
    font-weight: bold;
}

.content-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    border-left: 4px solid #007bff;
}

.generation-progress {
    background-color: #e3f2fd;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = None
    if 'content_router' not in st.session_state:
        st.session_state.content_router = ContentRouter()
    if 'prompt_builder' not in st.session_state:
        st.session_state.prompt_builder = SmartPromptBuilder()
    if 'generated_content_plan' not in st.session_state:
        st.session_state.generated_content_plan = None
    if 'routed_content' not in st.session_state:
        st.session_state.routed_content = None
    if 'generated_media' not in st.session_state:
        st.session_state.generated_media = {}
    if 'generation_status' not in st.session_state:
        st.session_state.generation_status = {}

def setup_model_manager():
    """Setup and test the model manager."""
    with st.spinner("Initializing AI models..."):
        try:
            # Get Hugging Face token from user input or environment
            hf_token = st.session_state.get('hf_token', os.getenv('HUGGINGFACE_TOKEN'))
            
            # Initialize model manager
            model_manager = HuggingFaceModelManager(hf_token)
            
            # Test model availability
            text_available = model_manager.test_model_availability('text')
            image_available = model_manager.test_model_availability('image')
            video_available = model_manager.test_model_availability('video')
            
            st.session_state.model_manager = model_manager
            
            # Display model status
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status = "✅ Available" if text_available else "❌ Unavailable"
                st.markdown(f"**Text Generation:** {status}")
            
            with col2:
                status = "✅ Available" if image_available else "❌ Unavailable"
                st.markdown(f"**Image Generation:** {status}")
            
            with col3:
                status = "✅ Available" if video_available else "❌ Unavailable"
                st.markdown(f"**Video Generation:** {status}")
            
            if not any([text_available, image_available, video_available]):
                st.error("No models are available. Please check your Hugging Face token and internet connection.")
                return False
            
            return True
            
        except Exception as e:
            st.error(f"Failed to initialize models: {str(e)}")
            st.info("You can still use the app in demo mode with placeholder content.")
            return False

def render_sidebar():
    """Render the sidebar with configuration options."""
    st.sidebar.markdown("### ⚙️ Configuration")
    
    # Hugging Face token input
    hf_token = st.sidebar.text_input(
        "Hugging Face Token (Optional)",
        type="password",
        help="Enter your Hugging Face token for authenticated access to models"
    )
    
    if hf_token:
        st.session_state.hf_token = hf_token
    
    # Model status
    st.sidebar.markdown("### 📊 Model Status")
    if st.session_state.model_manager:
        status = st.session_state.model_manager.model_status
        for model_type, info in status.items():
            icon = "✅" if info['available'] else "❌"
            st.sidebar.markdown(f"{icon} **{model_type.title()}:** {info.get('current_model', 'N/A')}")
    else:
        st.sidebar.info("Models not initialized")
    
    # App information
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        "This app uses agentic AI to generate content plans and automatically "
        "create multimedia content using Hugging Face models."
    )
    
    # Instructions
    st.sidebar.markdown("### 📋 Instructions")
    st.sidebar.markdown("""
1. Enter your client type (e.g., chef, accountant, artist)
2. Click 'Generate Content Plan' to create a content strategy
3. Review the generated plan
4. Generate multimedia content automatically
5. Download or view the generated content
    """)

def render_client_input():
    """Render the client input section."""
    st.markdown('<div class="section-header">📝 Client Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        client_type = st.text_input(
            "What type of client is this?",
            placeholder="e.g., chef, accountant, artist, dentist, fitness trainer...",
            help="Enter the profession or business type of your client"
        )
    
    with col2:
        duration = st.selectbox(
            "Content Duration",
            ["1 month", "2 weeks", "1 week", "3 months"],
            help="How long should the content plan cover?"
        )
    
    return client_type, duration

def generate_content_plan(client_type: str, duration: str):
    """Generate the content plan using the text generation model."""
    if not client_type:
        st.error("Please enter a client type first.")
        return
    
    st.markdown('<div class="section-header">🤖 Generating Content Plan</div>', unsafe_allow_html=True)
    
    with st.spinner(f"Creating {duration} content plan for {client_type}..."):
        try:
            if st.session_state.model_manager and st.session_state.model_manager.model_status['text']['available']:
                # Use actual AI model
                text_gen = TextGenerator(st.session_state.model_manager)
                content_plan = text_gen.generate_content_plan(client_type, duration)
            else:
                # Use fallback/demo content
                content_plan = generate_demo_content_plan(client_type, duration)
            
            st.session_state.generated_content_plan = content_plan
            
            # Parse and display the content plan
            try:
                parsed_plan = json.loads(content_plan)
                st.success(f"✅ Content plan generated successfully!")
                
                # Display summary
                days_count = len(parsed_plan.get('days', []))
                st.info(f"📅 Generated {days_count} days of content for {client_type}")
                
                # Display the content plan
                with st.expander("📋 View Generated Content Plan", expanded=True):
                    st.json(parsed_plan)
                
                return content_plan
                
            except json.JSONDecodeError:
                st.error("Generated content is not in valid JSON format. Please try again.")
                return None
                
        except Exception as e:
            st.error(f"Error generating content plan: {str(e)}")
            st.info("Using demo content plan instead.")
            demo_plan = generate_demo_content_plan(client_type, duration)
            st.session_state.generated_content_plan = demo_plan
            return demo_plan

def generate_demo_content_plan(client_type: str, duration: str) -> str:
    """Generate a demo content plan when models are not available."""
    demo_plan = {
        "client_type": client_type,
        "duration": duration,
        "days": [
            {
                "day": 1,
                "content_idea": f"Welcome post introducing {client_type} services",
                "infographic_description": f"Professional design showcasing {client_type} expertise with modern clean layout",
                "hashtags": f"#{client_type.lower().replace(' ', '')} #professional #services #welcome",
                "post_caption": f"Welcome to our {client_type} page! Excited to share our expertise with you.",
                "video_script": f"Quick introduction video showing {client_type} workspace and team"
            },
            {
                "day": 2,
                "content_idea": f"Tips and tricks for {client_type} industry",
                "infographic_description": f"Colorful infographic with 5 essential tips for {client_type} clients",
                "hashtags": f"#{client_type.lower().replace(' ', '')}tips #advice #professional #tips",
                "post_caption": f"Here are our top 5 tips for anyone working with a {client_type}!",
                "photo_description": f"Behind-the-scenes photo of {client_type} at work"
            },
            {
                "day": 3,
                "content_idea": f"Client success story featuring {client_type} work",
                "infographic_description": f"Before/after style infographic showing {client_type} project results",
                "hashtags": f"#{client_type.lower().replace(' ', '')} #success #clientstory #results",
                "post_caption": f"Amazing results from our recent {client_type} project! So proud of this outcome.",
                "video_script": f"Client testimonial video about working with {client_type}"
            }
        ]
    }
    
    return json.dumps(demo_plan, indent=2)

def route_and_generate_content(content_plan: str, client_type: str):
    """Route content and generate multimedia materials."""
    if not content_plan:
        st.error("No content plan available. Please generate a content plan first.")
        return
    
    st.markdown('<div class="section-header">🎯 Content Routing & Generation</div>', unsafe_allow_html=True)
    
    # Route the content
    with st.spinner("Analyzing and routing content..."):
        routed_content = st.session_state.content_router.analyze_content_plan(content_plan)
        st.session_state.routed_content = routed_content
    
    # Display routing summary
    routing_summary = st.session_state.content_router.get_routing_summary(routed_content)
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("**Routing Summary:**")
    st.text(routing_summary)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Generate content for each type
    if st.button("🚀 Generate All Media Content", type="primary"):
        generate_all_media_content(routed_content, client_type)

def generate_all_media_content(routed_content: Dict, client_type: str):
    """Generate all multimedia content based on routed content."""
    st.markdown('<div class="section-header">🎨 Multimedia Generation</div>', unsafe_allow_html=True)
    
    # Initialize generators
    if st.session_state.model_manager:
        image_gen = ImageGenerator(st.session_state.model_manager)
        video_gen = VideoGenerator(st.session_state.model_manager)
    else:
        image_gen = None
        video_gen = None
    
    # Create tabs for different content types
    tab1, tab2, tab3 = st.tabs(["📝 Text Content", "🖼️ Images", "🎥 Videos"])
    
    with tab1:
        display_text_content(routed_content.get('text_generation', []))
    
    with tab2:
        generate_and_display_images(routed_content.get('image_generation', []), client_type, image_gen)
    
    with tab3:
        generate_and_display_videos(routed_content.get('video_generation', []), client_type, video_gen)

def display_text_content(text_content: List[Dict]):
    """Display text content in organized format."""
    if not text_content:
        st.info("No text content to display.")
        return
    
    for i, item in enumerate(text_content):
        with st.expander(f"Day {item.get('day', i+1)} - {item.get('type', 'Content')}", expanded=False):
            st.markdown(f"**Content:** {item.get('content', 'N/A')}")

def generate_and_display_images(image_content: List[Dict], client_type: str, image_gen: Optional[ImageGenerator]):
    """Generate and display images."""
    if not image_content:
        st.info("No image content to generate.")
        return
    
    for i, item in enumerate(image_content):
        day = item.get('day', i+1)
        content_type = item.get('type', 'image')
        
        st.markdown(f"### Day {day} - {content_type}")
        
        # Build optimized prompt
        prompt = st.session_state.prompt_builder.build_image_prompt(item, client_type)
        st.markdown(f"**Prompt:** {prompt}")
        
        # Generate image
        with st.spinner(f"Generating image for Day {day}..."):
            try:
                if image_gen and st.session_state.model_manager.model_status['image']['available']:
                    # Use actual AI model
                    image = image_gen.generate_image(prompt)
                    if hasattr(image, 'save'):  # PIL Image
                        st.image(image, caption=f"Day {day} - {content_type}", use_column_width=True)
                    else:
                        st.success(f"Image generated: {image}")
                else:
                    # Show placeholder
                    st.info(f"🖼️ Image would be generated here for: {prompt[:100]}...")
                    st.markdown("*Demo mode: Actual image generation requires model availability*")
                    
            except Exception as e:
                st.error(f"Error generating image: {str(e)}")
        
        st.markdown("---")

def generate_and_display_videos(video_content: List[Dict], client_type: str, video_gen: Optional[VideoGenerator]):
    """Generate and display videos."""
    if not video_content:
        st.info("No video content to generate.")
        return
    
    for i, item in enumerate(video_content):
        day = item.get('day', i+1)
        content_type = item.get('type', 'video')
        
        st.markdown(f"### Day {day} - {content_type}")
        
        # Build optimized prompt
        prompt = st.session_state.prompt_builder.build_video_prompt(item, client_type)
        st.markdown(f"**Prompt:** {prompt}")
        
        # Generate video
        with st.spinner(f"Processing video for Day {day}..."):
            try:
                if video_gen and st.session_state.model_manager.model_status['video']['available']:
                    # Use actual AI model
                    result = video_gen.generate_video(prompt)
                    st.success(f"Video processing: {result}")
                else:
                    # Show placeholder
                    st.info(f"🎥 Video would be generated here for: {prompt[:100]}...")
                    st.markdown("*Demo mode: Actual video generation requires model availability and significant computational resources*")
                    
            except Exception as e:
                st.error(f"Error generating video: {str(e)}")
        
        st.markdown("---")

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🎨 Context Creation App with Agentic AI</div>', unsafe_allow_html=True)
    st.markdown("Generate comprehensive content plans and multimedia content for any client type using advanced AI models.")
    
    # Render sidebar
    render_sidebar()
    
    # Initialize models if not already done
    if st.session_state.model_manager is None:
        if st.button("🚀 Initialize AI Models"):
            setup_model_manager()
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Client input section
        client_type, duration = render_client_input()
        
        # Generate content plan button
        if st.button("📋 Generate Content Plan", type="primary"):
            if client_type:
                generate_content_plan(client_type, duration)
            else:
                st.error("Please enter a client type first.")
    
    with col2:
        # Content generation section
        if st.session_state.generated_content_plan:
            route_and_generate_content(st.session_state.generated_content_plan, client_type)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and Hugging Face models | MiniMax Agent")

if __name__ == "__main__":
    main()

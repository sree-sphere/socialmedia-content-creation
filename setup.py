"""
Setup script for Context Creation App with Agentic AI
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def setup_environment():
    """Setup environment variables and configuration."""
    print("⚙️ Setting up environment...")
    
    # Create .env file template if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# Hugging Face Configuration
HUGGINGFACE_TOKEN=your_token_here

# Optional: Model Configuration
# You can specify preferred models here
DEFAULT_TEXT_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
DEFAULT_IMAGE_MODEL=stabilityai/stable-diffusion-xl-base-1.0
DEFAULT_VIDEO_MODEL=ali-vilab/modelscope-damo-text-to-video-synthesis
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("📄 Created .env file template")
    
    # Create output directories
    dirs_to_create = [
        "generated_content",
        "generated_content/images",
        "generated_content/videos",
        "generated_content/plans"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("📁 Created output directories")

def run_app():
    """Run the Streamlit app."""
    print("🚀 Starting the Context Creation App...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "code/streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

def main():
    """Main setup function."""
    print("🎨 Context Creation App Setup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("code/streamlit_app.py").exists():
        print("❌ Please run this script from the root directory of the project")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Ask user if they want to run the app
    print("\n🎉 Setup complete!")
    print("\n📋 Next steps:")
    print("1. Edit .env file and add your Hugging Face token")
    print("2. Run the app with: python setup.py --run")
    print("   or directly with: streamlit run code/streamlit_app.py")
    
    if "--run" in sys.argv:
        run_app()

if __name__ == "__main__":
    main()

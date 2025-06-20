# 🎨 Context Creation App with Agentic AI

This uses agentic AI to generate content plans and automatically create multimedia content using Hugging Face models.

## 🌟 Features

### Core Functionality
- **Intelligent Content Planning**: Generate day-wise content strategies for any client type
- **Agentic AI Routing**: Automatically route different content types to appropriate AI models
- **Multi-Modal Generation**: Create text, images, and videos from content plans

### AI Models Integration
- **Text Generation**: Llama 3, Mixtral 8x7B, Flan-T5
- **Image Generation**: Stable Diffusion XL, Stable Diffusion v1.5
- **Video Generation**: ModelScope T2V, ZeroScope v2, Stable Video Diffusion

## Quick Start

### Installation

1. **Clone or download the project files**
   ```bash
   # Ensure you have all the code files in your workspace
   ```

2. **Run the setup script**
   ```bash
   python setup.py
   ```

3. **Configure your environment**
   - Edit the `.env` file and add your Hugging Face token:
     ```
     HUGGINGFACE_TOKEN=token_here
     ```

4. **Launch the application**
   ```bash
   python setup.py --run
   ```
   
   Or manually:
   ```bash
   streamlit run code/streamlit_app.py
   ```

## 📋 Usage Guide

### Step 1: Initialize Models
1. Open the app in your browser
2. Enter your Hugging Face token in the sidebar
3. Click "Initialize AI Models" to test model availability

### Step 2: Generate Content Plan
1. Enter your client type ("chef", "accountant", "fitness trainer")
2. Select content duration (1 week to 3 months)
3. Click "Generate Content Plan"
4. Review the generated JSON content plan

### Step 3: Generate Multimedia Content
1. The app automatically analyzes and routes content
2. Click "Generate All Media Content"
3. Review generated images, videos, and text content in the tabs

### Step 4: Download and Use
- Generated content is displayed in the app
- Images and videos are saved to the `generated_content/` directory
- Content plans can be exported as JSON files

## Architecture

### Agentic AI System
The app uses a **Router/Dispatcher pattern** for intelligent content routing:

```
User Input → Content Plan Generation → Agentic Router → Specialized Models
                                            ↓
                Text ← Image ← Video ← Content Classification
```

### Key Components

#### 1. Content Router (`agentic_content_router.py`)
- Analyzes JSON content plans
- Classifies content types using keyword matching
- Routes content to appropriate generation models
- Builds optimized prompts for each model type

#### 2. Model Manager (`huggingface_models.py`)
- Manages Hugging Face model connections
- Handles model availability testing
- Provides unified interfaces for different model types
- Implements fallback mechanisms

## Configuration

### Model Configuration
The app automatically selects optimal models based on availability:

| Model Type | Primary | Fallback | Purpose |
|------------|---------|----------|---------|
| Text | Llama 3 8B | DialoGPT | Content planning, JSON generation |
| Image | Stable Diffusion XL | SD v1.5 | Infographics, social media visuals |
| Video | ModelScope T2V | ZeroScope v2 | Promotional videos, tutorials |

## Project Structure

```
context-creation-app/
├── code/
│   ├── streamlit_app.py           # Main Streamlit application
│   ├── agentic_content_router.py  # Content routing and classification
│   └── huggingface_models.py      # Model integrations
├── docs/
│   └── research_report_content_creation_app.md
├── generated_content/             # Output directory
│   ├── images/
│   ├── videos/
│   └── plans/
├── requirements.txt               # Python dependencies
├── setup.py                      # Setup and installation script
├── .env                          # Environment configuration
└── README.md                     # This file
```

## Example Use Cases

### 1. Restaurant Owner (Chef)
**Input**: "chef"
**Generated Plan**: 
- Day 1: Welcome post with signature dish infographic
- Day 2: Behind-the-scenes kitchen video
- Day 3: Recipe tip with step-by-step visuals

### 2. Accounting Firm
**Input**: "accountant"
**Generated Plan**:
- Day 1: Tax season tips infographic
- Day 2: Client success story with charts
- Day 3: Educational video on bookkeeping

### 3. Fitness Trainer
**Input**: "fitness trainer"
**Generated Plan**:
- Day 1: Workout routine infographic
- Day 2: Client transformation video
- Day 3: Nutrition tips with meal prep visuals

## Advanced Usage

### Custom Model Integration
To add new Hugging Face models:

1. Edit `huggingface_models.py`
2. Add model names to the `models` configuration
3. Update the model testing methods if needed

### Extending Content Types
To add new content classification:

1. Edit `agentic_content_router.py`
2. Add new keywords to the classification system
3. Create new routing rules in `_classify_content()`

### Custom Prompt Templates
Modify prompt templates in `SmartPromptBuilder` class:

```python
self.image_prompt_templates = {
    'your_type': "Your custom prompt template for {client_type}: {content}"
}
```

## Troubleshooting

### Common Issues

#### Model Not Available
- **Cause**: Network issues, invalid token, or model capacity limits
- **Solution**: Check internet connection, verify Hugging Face token, try fallback models

#### JSON Parsing Errors
- **Cause**: Model generated invalid JSON format
- **Solution**: App automatically uses fallback content plans

#### Slow Performance
- **Cause**: Large models or high demand
- **Solution**: Use smaller models or run during off-peak hours

#### Memory Issues
- **Cause**: Video generation requires significant resources
- **Solution**: Close other applications, use cloud computing instances

### Debug Mode
Enable debug logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Standards
- Follow PEP 8 for Python code
- Add docstrings to all functions
- Include type hints where possible
- Write comprehensive error handling

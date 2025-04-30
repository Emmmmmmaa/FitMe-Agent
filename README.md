<!-- ---
title: FitMe Agent
emoji: 👗
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 3.50.0
app_file: src/app.py
pinned: false
license: mit
short_description: An intelligent fashion recommendation agent
--- -->

# AI Fashion Recommendation System

<!-- An intelligent fashion recommendation system built with Microsoft's AI technologies. -->
An intelligent fashion recommendation agent powered by image segmentation, multi-modal fusion, and large language models (LLMs). It analyzes Taobao purchase history to generate personalized outfit recommendations with detailed explanations.

YouTube Illustration
https://github.com/Emmmmmmaa/FitMe-Agent![image](https://github.com/user-attachments/assets/ec1689f5-5440-4d51-9bf5-3e7df106da93)

## Features

- Image upload and processing
- Style scene selection
- Clothing image segmentation
- Color analysis and style matching
- Look recommendation with rule-based engine
- Natural language recommendations using GPT-4

## Technology Stack

- Semantic Kernel for orchestration
- Autogen for collaborative agents
- Azure AI Agents SDK
- Microsoft 365 Agents SDK
- Gradio for UI
- Azure GPT-4 API

## Project Structure

```
├── src/
│   ├── agents/           # AI agents implementation
│   ├── models/           # ML models and processing
│   ├── rules/            # Fashion rules engine
│   ├── utils/            # Utility functions
│   └── app.py            # Main application
├── tests/                # Test files
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Azure credentials
```

3. Run the application:
```bash
python src/app.py
```

## License

MIT 


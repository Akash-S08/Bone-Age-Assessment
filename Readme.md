🦴 Bone Age Assessment System

An AI-powered Bone Age Assessment System that predicts bone age from hand X-ray images using Deep Learning and provides visual explanations using Grad-CAM.

The project uses a trained PyTorch deep learning model to analyze hand X-ray images, predict bone age, and highlight the important regions that influenced the prediction.

📌 Project Overview

Bone age assessment is a medical technique used to evaluate the skeletal maturity and development of an individual. Traditionally, specialists analyze hand and wrist X-ray images to estimate bone age.

This project aims to assist and automate this process using Deep Learning, Computer Vision, and Explainable AI.

The system can:

Accept hand X-ray images
Preprocess images for the deep learning model
Predict the estimated bone age
Generate Grad-CAM visualizations
Highlight important regions used by the model
Provide predictions through a FastAPI backend
Display results through a web-based frontend
✨ Features
🦴 Bone age prediction from hand X-ray images
🤖 Deep Learning-based prediction using PyTorch
🔍 Explainable AI using Grad-CAM
🖼️ Image preprocessing using OpenCV and Pillow
🔄 Data augmentation using Albumentations
🚀 FastAPI backend for model predictions
🌐 Web-based frontend using HTML, CSS, and JavaScript
📊 Training and validation pipeline
💾 Saves the best and latest trained models
🧪 Includes pipeline testing
📈 Visualization and analysis support
🛠️ Technologies Used
🧠 Artificial Intelligence & Deep Learning
Python
PyTorch
TorchVision
⚙️ Backend
FastAPI
Uvicorn
🖼️ Computer Vision & Image Processing
OpenCV
Pillow
Albumentations
📊 Data Processing & Machine Learning
NumPy
Pandas
Scikit-learn
📈 Visualization
Matplotlib
Seaborn
Grad-CAM
🌐 Frontend
HTML
CSS
JavaScript
🧪 Testing
Pytest
📂 Project Structure
bone_age_assessment/
│
├── backend/
│   ├── __pycache__/          # Python cache files
│   ├── api.py                # FastAPI backend application
│   ├── config.json           # Backend configuration
│   ├── dataset.py            # Dataset loading and preprocessing
│   ├── model.py              # Deep learning model architecture
│   ├── train.py              # Model training script
│   └── utils.py              # Utility functions
│
├── data/
│   ├── train/                # Training X-ray images
│   ├── val/                  # Validation X-ray images
│   ├── train_labels.csv      # Training labels
│   └── val_labels.csv        # Validation labels
│
├── frontend/
│   ├── app.js                # Frontend JavaScript
│   ├── index.html            # Main web page
│   └── styles.css            # Frontend styling
│
├── logs/                     # Application and training logs
│
├── models/
│   ├── best_model.pth        # Best trained model
│   └── latest_model.pth      # Latest trained model
│
├── tests/
│   ├── php.php               # Additional test file
│   └── test_pipeline.py      # Pipeline testing
│
├── venv/                     # Python virtual environment
│
├── config.json               # Project configuration
├── requirements.txt          # Required Python packages
└── README.md                 # Project documentation
🔄 System Workflow
                 Hand X-Ray Image
                        │
                        ▼
              Image Preprocessing
                        │
                        ▼
               Data Augmentation
                        │
                        ▼
            PyTorch Deep Learning Model
                        │
                        ▼
               Bone Age Prediction
                        │
                        ▼
              Grad-CAM Visualization
                        │
                        ▼
               FastAPI Backend
                        │
                        ▼
                Web Frontend Display
🧠 How the System Works
1️⃣ Image Input

The user provides a hand X-ray image through the application.

2️⃣ Image Preprocessing

The image is processed before being passed to the deep learning model.

This may include:

Resizing
Normalization
Image conversion
Data augmentation
3️⃣ Deep Learning Model

The processed image is passed to the trained PyTorch model.

The model analyzes skeletal features in the X-ray image and predicts the estimated bone age.

4️⃣ Bone Age Prediction

The trained model generates a predicted bone age based on the skeletal development visible in the X-ray image.

5️⃣ Grad-CAM Visualization

The project uses Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize the important regions of the X-ray image that influenced the model's prediction.

Grad-CAM improves the interpretability of the AI model by providing a visual explanation of where the model focused.

6️⃣ Result Display

The prediction and visualization are sent through the FastAPI backend and displayed on the frontend.

🔍 Explainable AI with Grad-CAM

Grad-CAM is used to make the deep learning model more interpretable.

Instead of only providing a bone age prediction, Grad-CAM generates a visualization that highlights important regions of the hand X-ray image.

This helps users understand:

Which areas influenced the prediction
Where the deep learning model focused
How the model analyzed the X-ray image
Benefits of Grad-CAM
🔍 Improves model interpretability
🧠 Provides visual explanations
📊 Highlights important image regions
🤖 Makes AI predictions easier to understand
🏥 Supports explainable AI in medical imaging
⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/YOUR-USERNAME/bone_age_assessment.git
2️⃣ Navigate to the Project Directory
cd bone_age_assessment
3️⃣ Create a Virtual Environment
python -m venv venv
4️⃣ Activate the Virtual Environment
Windows
venv\Scripts\activate
Linux/macOS
source venv/bin/activate
5️⃣ Install Dependencies
pip install -r requirements.txt
📦 Dependencies

The project uses the following libraries:

PyTorch
TorchVision
FastAPI
Uvicorn
OpenCV
Pillow
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
Albumentations
Python-Multipart
Jinja2
Aiofiles
Pytest

All dependencies can be installed using:

pip install -r requirements.txt
🏋️ Model Training

The training logic is located in:

backend/train.py

To train the model, run:

python backend/train.py

During training, the system:

Loads the training dataset
Applies preprocessing and augmentation
Trains the deep learning model
Evaluates the model using validation data
Saves model checkpoints

The trained models are stored in:

models/
💾 Model Files

The project saves trained models inside the models directory.

best_model.pth

Contains the model checkpoint with the best performance during training.

latest_model.pth

Contains the most recently saved model checkpoint.

🚀 Running the Application

Start the FastAPI backend using:

uvicorn backend.api:app --reload

Once the server is running, open the application in your browser.

By default, FastAPI applications commonly run on:

http://127.0.0.1:8000
📚 API Documentation

FastAPI automatically provides interactive API documentation.

After starting the backend, access the API documentation at:

http://127.0.0.1:8000/docs

The documentation allows you to test and explore the available API endpoints.

🌐 Frontend

The frontend is built using:

HTML
CSS
JavaScript

The frontend files are located inside:

frontend/
├── index.html
├── styles.css
└── app.js

The frontend communicates with the FastAPI backend to send X-ray images and display prediction results.

📊 Dataset

The dataset is organized into training and validation sections.

data/
│
├── train/
│   └── Training X-ray Images
│
├── val/
│   └── Validation X-ray Images
│
├── train_labels.csv
│
└── val_labels.csv

The CSV files contain the labels associated with the corresponding X-ray images.

🧪 Testing

The project includes testing functionality inside the tests directory.

Run all tests using:

pytest

Or run the pipeline test directly:

python tests/test_pipeline.py

⚠️ Disclaimer

This project is developed for educational and research purposes only.

The predictions generated by this system should not be considered a replacement for professional medical diagnosis, treatment, or advice.

Medical decisions should always be made by qualified healthcare professionals.

👨‍💻 Author

Akash

Recent Computer Science and Engineering Graduate interested in:

💻 Software Development
🧠 Machine Learning
🔬 Deep Learning
⚙️ Backend Development
⭐ Support

If you found this project interesting or useful, please consider giving the repository a ⭐ star!
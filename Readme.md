🦴 Bone Age Assessment System

An AI-powered Bone Age Assessment System that predicts bone age from hand X-ray images using Deep Learning and provides visual explanations using Grad-CAM.

The project uses a trained PyTorch deep learning model to analyze hand X-ray images, predict bone age, and highlight important regions that influenced the prediction.

📌 Project Overview

Bone age assessment is used to evaluate the skeletal maturity and development of an individual. Traditionally, specialists analyze hand and wrist X-ray images to estimate bone age.

This project aims to assist this process using Deep Learning, Computer Vision, and Explainable AI.

The system can:

Accept hand X-ray images

Preprocess images for the deep learning model
Predict the estimated bone age
Generate Grad-CAM visualizations
Highlight important regions that influenced the prediction
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
💾 Saves trained model checkpoints
🧪 Includes pipeline testing
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

📊 Data Processing

NumPy
Pandas
Scikit-learn
📈 Visualization & Explainable AI
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

Bone-Age-Assessment/
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
│   └── test_pipeline.py      # Pipeline testing
│
├── config.json               # Project configuration
├── requirements.txt          # Required Python packages
├── .gitignore                # Files ignored by Git
└── README.md                 # Project documentation

Note: The venv/ and __pycache__/ folders should not be uploaded to GitHub.

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

The user uploads a hand X-ray image through the web application.

2️⃣ Image Preprocessing

The image is prepared before being passed to the deep learning model. This can include:

Resizing
Normalization
Image conversion
Data augmentation
3️⃣ Deep Learning Model

The processed image is passed to the trained PyTorch model.

The model analyzes skeletal features visible in the X-ray image and predicts the estimated bone age.

4️⃣ Bone Age Prediction

The trained model generates a predicted bone age based on patterns of skeletal development identified in the hand X-ray image.

5️⃣ Grad-CAM Visualization

The project uses Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize important regions of the X-ray image that influenced the model's prediction.

This provides an additional layer of interpretability by showing where the model focused while making its prediction.

6️⃣ Result Display

The prediction and Grad-CAM visualization are processed through the backend and displayed to the user through the frontend.

🔍 Explainable AI with Grad-CAM

Grad-CAM is used to make the deep learning model more interpretable.

Instead of only providing a bone age prediction, Grad-CAM generates a visual representation highlighting important regions of the hand X-ray image.

Benefits of Grad-CAM
🔍 Highlights important regions used by the model
🧠 Improves model interpretability
📊 Provides visual explanations for predictions
🤖 Makes AI decision-making easier to understand
🏥 Supports Explainable AI in medical imaging

⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/Akash-S08/Bone-Age-Assessment.git
2️⃣ Navigate to the Project Directory
cd Bone-Age-Assessment
3️⃣ Create a Virtual Environment
python -m venv venv
4️⃣ Activate the Virtual Environment
Windows
venv\Scripts\activate
Linux/macOS
source venv/bin/activate
5️⃣ Install Dependencies
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

The trained models are stored inside the models directory.

best_model.pth

Contains the model checkpoint with the best performance during training.

latest_model.pth

Contains the most recently saved model checkpoint.

🚀 Running the Application

Start the FastAPI backend using:

uvicorn backend.api:app --reload

Once the server is running, open the application in your browser.

The application will typically be available at:

http://127.0.0.1:8000
📚 API Documentation

FastAPI provides interactive API documentation.

After starting the backend, visit:

http://127.0.0.1:8000/docs

You can use this page to view and test the available API endpoints.

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

The frontend communicates with the FastAPI backend to send X-ray images and display prediction results and Grad-CAM visualizations.

📊 Dataset Structure

The dataset is organized into training and validation sections.

data/
│
├── train/
│   └── Training X-ray images
│
├── val/
│   └── Validation X-ray images
│
├── train_labels.csv
│
└── val_labels.csv

The CSV files contain labels associated with the corresponding X-ray images.

🧪 Testing

The project includes testing functionality inside the tests directory.

Run all tests using:

pytest

Or run the pipeline test directly:

python tests/test_pipeline.py
📈 Project Components
Component	Description
dataset.py	Handles dataset loading and preprocessing
model.py	Contains the deep learning model architecture
train.py	Handles model training and validation
utils.py	Contains helper and utility functions
api.py		Provides FastAPI backend services
app.js		Handles frontend interactions
index.html	Defines the web application structure
styles.css	Provides frontend styling
Grad-CAM	Generates visual explanations for predictions

🔮 Future Improvements
Improve model prediction accuracy
Train with a larger dataset
Experiment with advanced deep learning architectures
Add prediction confidence scores
Improve Grad-CAM visualizations
Add user authentication
Store prediction history in a database
Deploy the application to the cloud
Create a mobile-friendly interface
Improve API security and scalability

⚠️ Disclaimer

This project is developed for educational and research purposes only.

The predictions generated by this system should not be considered a replacement for professional medical diagnosis, treatment, or advice.

Medical decisions should always be made by qualified healthcare professionals.

👨‍💻 Author

Akash

Recent Computer Science and Engineering Graduate interested in:

💻 Software Development
🤖 Artificial Intelligence
🧠 Machine Learning
🔬 Deep Learning
⚙️ Backend Development
🔗 Connect With Me
GitHub: Akash-S08
Project Repository: Bone-Age-Assessment

⭐ Support

If you found this project interesting or useful, please consider giving the repository a ⭐ star!

🏷️ Technologies & Topics

Python • PyTorch • Deep Learning • Machine Learning • Computer Vision • FastAPI • Grad-CAM • Explainable AI • Medical Imaging

📄 License

This project is currently intended for educational and learning purposes.
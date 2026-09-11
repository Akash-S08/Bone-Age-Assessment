🦴 Bone Age Assessment System

An AI-powered Bone Age Assessment System that uses Deep Learning to estimate bone age from pediatric hand X-ray images. The project uses **ResNet34 with CBAM attention mechanisms** for feature extraction and prediction, along with **Grad-CAM** to provide visual explanations of the model's decisions.

📌 Project Overview

Bone age assessment is used to evaluate skeletal maturity and development, traditionally requiring specialists to analyze hand and wrist X-ray images.

This project explores how **Artificial Intelligence and Computer Vision** can assist in this process by developing a deep learning-based system for estimating bone age from pediatric hand X-ray images.

The system combines **Deep Learning, Attention Mechanisms, and Explainable AI** to provide both predictions and visual insights into the regions that influenced the model's decision.


 ✨ Features

- 🦴 Bone age estimation from pediatric hand X-ray images
- 🧠 Deep learning-based image analysis using PyTorch
- 🔬 ResNet34 architecture for feature extraction
- 🎯 CBAM attention mechanism to enhance important features
- 🔍 Grad-CAM for Explainable AI and visual interpretation
- 🖼️ Image preprocessing and augmentation
- 🚀 FastAPI backend for serving predictions
- 🌐 Interactive frontend using HTML, CSS, and JavaScript
- 📊 Training and validation pipeline
- 💾 Model checkpoint saving
- 🧪 Pipeline testing



 🛠️ Technologies Used

 🧠 Deep Learning & AI

- Python
- PyTorch
- TorchVision
- ResNet34
- CBAM (Convolutional Block Attention Module)

 🔍 Explainable AI

- Grad-CAM

🖼️ Computer Vision & Image Processing

- OpenCV
- Pillow
- Albumentations

⚙️ Backend

- FastAPI
- Uvicorn

🌐 Frontend

- HTML
- CSS
- JavaScript

📊 Data Processing & Visualization

- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

📂 Project Structure

Bone-Age-Assessment/
│
├── backend/
│   ├── api.py                # FastAPI backend application
│   ├── config.json           # Backend configuration
│   ├── dataset.py            # Dataset loading and preprocessing
│   ├── model.py              # ResNet34 + CBAM model architecture
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
├── screenshots/              # Application screenshots
│
├── tests/
│   └── test_pipeline.py      # Pipeline testing
│
├── config.json               # Project configuration
├── requirements.txt          # Required Python packages
├── .gitignore                # Git ignored files
└── README.md                 # Project documentation

🔄 System Workflow
             Pediatric Hand X-Ray
                     │
                     ▼
            Image Preprocessing
                     │
                     ▼
             Data Augmentation
                     │
                     ▼
              ResNet34 + CBAM
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
               Web Application

🧠 Model Architecture

The system uses ResNet34 as the primary CNN architecture for extracting meaningful features from hand X-ray images.

To improve feature selection, the model incorporates CBAM (Convolutional Block Attention Module).

CBAM helps the network focus on important information by applying:

Channel Attention
Spatial Attention

This allows the model to emphasize relevant skeletal features that contribute to bone age estimation.

🔍 Explainable AI with Grad-CAM

The project uses Grad-CAM (Gradient-weighted Class Activation Mapping) to provide visual explanations for model predictions.

Grad-CAM generates a heatmap highlighting important regions of the hand X-ray that influenced the model's prediction.

This improves the interpretability of the system by allowing users to understand where the model focused during the prediction process.

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

The training script is located at:

backend/train.py

Run the training process using:

python backend/train.py

During training, the system:

Loads the training X-ray images
Applies preprocessing and augmentation
Extracts features using ResNet34
Applies CBAM attention mechanisms
Trains the model
Evaluates performance using validation data
Saves model checkpoints
💾 Model Files

Trained models are stored inside:

models/
best_model.pth

Contains the model checkpoint with the best validation performance.

latest_model.pth

Contains the most recently saved model checkpoint.

🚀 Running the Application

Start the FastAPI backend using:

uvicorn backend.api:app --reload

Once the server starts, open the application in your browser.

The application will typically run at:

http://127.0.0.1:8000

📚 API Documentation

FastAPI provides interactive API documentation.

After starting the backend, visit:

http://127.0.0.1:8000/docs

You can use this page to explore and test the available API endpoints.

🌐 Frontend

The frontend is built using:

HTML
CSS
JavaScript

Users can interact with the application by uploading a hand X-ray image and viewing:

The estimated bone age
Model prediction results
Grad-CAM visualization

📊 Dataset

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

The CSV files contain the labels associated with the corresponding X-ray images.

🧪 Testing

The project includes testing functionality inside the tests directory.

Run all tests using:

pytest

Or run the pipeline test directly:

python tests/test_pipeline.py

🔮 Future Improvements

Improve prediction accuracy with additional training data
Experiment with more advanced architectures
Improve CBAM attention mechanisms
Add prediction confidence scores
Enhance Grad-CAM visualizations
Store prediction history in a database
Add user authentication
Deploy the application to the cloud
Improve API scalability and security
Create a mobile-friendly interface

⚠️ Disclaimer

This project is developed for educational and research purposes only.

The predictions generated by this system should not be considered a replacement for professional medical diagnosis or advice.

Medical decisions should always be made by qualified healthcare professionals.

👨‍💻 Author

Akash

Recent Computer Science and Engineering Graduate interested in:

💻 Software Development
🤖 Artificial Intelligence
🧠 Machine Learning
🔬 Deep Learning
👁️ Computer Vision

🔗 Connect With Me
GitHub Profile — Akash-S08
Project Repository — Bone-Age-Assessment

⭐ Support

If you found this project interesting, consider giving the repository a ⭐ star!
# 🔢 Digit Recognizer Web App

A full-stack, deep learning application that recognizes handwritten digits (0-9). Built with a **FastAPI** backend running a **Keras/TensorFlow CNN model**, and a modern **Vanilla JS** frontend with high-performance canvas interaction.

## 🚀 Live Demo
- **Frontend**: [https://Kiiirtan.github.io/digit-recognizer-Web](https://kiiirtan.github.io/digit-recognizer-Web/frontend/index.html)
- **API (Backend)**: [https://digit-recognizer-api-eukx.onrender.com](https://digit-recognizer-api-eukx.onrender.com)

---

## 🏗️ Architecture

### 🧠 Backend (Deep Learning API)
- **Framework**: FastAPI
- **Model**: Convolutional Neural Network (CNN) trained on the MNIST dataset.
- **Libraries**: TensorFlow, Keras, NumPy, OpenCV.
- **Deployment**: Dockerized and hosted on **Render**.

### 🎨 Frontend (Client Interface)
- **Framework**: Vanilla JavaScript, Tailwind CSS, GSAP (Animations).
- **Key Features**: 
  - **Interactive Canvas**: Custom drawing logic with mobile touch support.
  - **Image Preprocessing**: Client-side bounding box extraction, aspect-ratio-aware scaling, and normalization before sending to the API.
  - **Image Upload**: Supports uploading external images (PNG/JPG) with dynamic inversion for dark/light backgrounds.

---

## 🛠️ Installation & Local Development

### Prerequisites
- Python 3.9+
- Node.js (optional, for local frontend serving)

### Backend Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Kiiirtan/digit-recognizer-Web.git
   cd digit-recognizer-Web
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the server:
   ```bash
   uvicorn backend.app:app --reload
   ```

### Frontend Setup
Simply open `frontend/index.html` in your browser. The code automatically detects if you are running locally and switches to `127.0.0.1:8000` for API calls.

---

## 🔮 Future Roadmap (Suggestions)

### 1. Client-Side Inference (TensorFlow.js)
Convert the `.keras` model to **TensorFlow.js** to run predictions directly in the browser.
- **Benefit**: Removes server latency and eliminates "cold start" issues on free hosting tiers.

### 2. Active Learning Loop
Add a feedback button (e.g., "Was this correct?") to collect misclassifications.
- **Benefit**: Build a custom dataset of edge cases to improve the model's accuracy in future training rounds.

### 3. Multi-Digit Support
Implement **Object Detection** or **Segmentation** (e.g., using OpenCV) to recognize multiple digits in a single drawing/image.

### 4. Advanced Preprocessing
Add **Gaussian Blurring** and **Rotation Normalization** to the preprocessing pipeline to make the model more robust to noisy or tilted inputs.

### 5. Mobile Progressive Web App (PWA)
Add a manifest and service worker to allow users to "install" the app on their mobile home screens and use it offline.

---

## 👨‍💻 Authors
- **Kirtan Patidar**
- **Harshit Walke**
- *B.Tech CSE • AI & Data Science*

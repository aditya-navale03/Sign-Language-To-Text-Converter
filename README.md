# Live Site: https://sign-language-converter.vercel.app/

# Sign Language Translator

A real-time sign language translator that uses computer vision to interpret sign language and display the translated text.

## Features

- Real-time camera feed for sign language input
- WebSocket-based communication between frontend and backend
- Clean and responsive UI
- Built with React and Python (Flask + OpenCV + MediaPipe)

## Prerequisites

- Node.js (v14+)
- Python (3.7+)
- pip (Python package manager)

## Setup Instructions

### Backend (Python)

1. Navigate to the PythonBackend directory:
   ```bash
   cd PythonBackend
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the Python backend server:
   ```bash
   python app.py
   ```
   The backend will start on `http://localhost:5000`

### Frontend (React)

1. In a new terminal, navigate to the Frontend directory:
   ```bash
   cd Frontend
   ```

2. Install the required Node.js packages:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```
   The frontend will open in your default browser at `http://localhost:3000`

## Usage

1. Click the "Start Camera" button to enable your webcam
2. Once the camera is active, click "Start Translation" to begin sign language recognition
3. Perform sign language gestures in front of your camera
4. The translated text will appear in the right panel
5. Click "Stop Translation" to pause the recognition

## Project Structure

```
sign-language-converter/
├── Frontend/               # React frontend
│   ├── public/             # Static files
│   └── src/                # React source code
│       ├── App.js          # Main React component
│       └── App.css         # Styling
├── PythonBackend/          # Python backend
│   ├── app.py             # Flask server and image processing
│   └── requirements.txt    # Python dependencies
└── README.md              # This file
```

## Technologies Used

- **Frontend**: React, WebSocket API
- **Backend**: Python, Flask, OpenCV, MediaPipe
- **Communication**: WebSocket for real-time data transfer

## License

This project is open source and available under the MIT License.

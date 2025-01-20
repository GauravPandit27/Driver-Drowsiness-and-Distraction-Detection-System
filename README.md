
---

# **Driver Drowsiness and Distraction Detection System**

This project uses advanced computer vision techniques to detect driver drowsiness and distractions, helping to enhance road safety by providing timely warnings.

---

## **Table of Contents**
1. [Features](#features)
2. [Technologies Used](#technologies-used)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Functionality](#functionality)
6. [Project Structure](#project-structure)
7. [Demo](#demo)
8. [License](#license)

---

## **Features**
- **Drowsiness Detection**: Tracks eye aspect ratio (EAR) to monitor eye closure and detect signs of drowsiness.
- **Phone Detection**: Identifies if the driver is using a phone, a major source of distraction.
- **Flashing Alert**: Activates a flashing screen for visual alerts when drowsiness or distraction is detected.
- **Voice Warnings**: Uses Text-to-Speech (TTS) for auditory warnings to alert the driver.
- **Real-Time Monitoring**: Processes video feed from the webcam in real time.
- **Configurable Thresholds**: Allows easy tuning of EAR thresholds and alert intervals.

---

## **Technologies Used**
- **Python**: Core programming language.
- **OpenCV**: For video processing and face/eye detection.
- **Mediapipe**: For facial landmarks detection.
- **YOLOv8**: For real-time object detection (phone detection).
- **pyttsx3**: For Text-to-Speech warnings.
- **NumPy**: For numerical operations.
- **Time**: For timing-related logic.

---

## **Installation**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/driver-drowsiness-distraction.git
   cd driver-drowsiness-distraction
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv8 Weights**
   - Download the YOLOv8 weights (`yolov8n.pt`) from the [Ultralytics YOLOv8 repository](https://github.com/ultralytics/ultralytics) and place it in the project directory.

---

## **Usage**

1. **Run the Script**
   ```bash
   python main.py
   ```

2. **How It Works**:
   - A webcam feed will open.
   - The system monitors your eyes for drowsiness and detects phone usage.
   - Alerts (visual flashing and TTS warnings) will be triggered based on predefined thresholds.

3. **Adjust Configurations**:
   - You can modify thresholds and parameters (e.g., EAR threshold, alert interval) in the `main.py` file.

---

## **Functionality**

### 1. **Drowsiness Detection**
   - Uses the Eye Aspect Ratio (EAR) to calculate eye closure.
   - If the eyes remain closed for a continuous period, it triggers a warning.

### 2. **Phone Detection**
   - Uses YOLOv8 for detecting phones in the video feed.
   - If a phone is detected in the frame, it issues a distraction warning.

### 3. **Flashing Screen**
   - Provides a visual alert by toggling a full-screen flash.

### 4. **Text-to-Speech Warnings**
   - Alerts the driver audibly with predefined warning messages.

---

## **Project Structure**

```plaintext
driver-drowsiness-distraction/
├── main.py               # Main script for running the detection system
├── requirements.txt      # Dependencies required for the project
├── README.md             # Project documentation
├── yolov8n.pt            # YOLOv8 model weights



---
```

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Contributing**
We welcome contributions to enhance the project! Feel free to submit issues or pull requests.

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed explanation.

---

If you have any questions or feedback, please feel free to reach out!

--- 

✋ Hand Gesture Math Solver

A real-time computer vision project that allows users to solve mathematical expressions using hand gestures. The system detects hand movements through a webcam and interprets them as numbers and mathematical operations.

Built using Python, OpenCV, and MediaPipe.

🚀 Features
✅ Real-time hand detection
✅ Finger counting recognition
✅ Gesture-based number input
✅ Mathematical operation recognition (+, -, ×, ÷)
✅ Live expression display on screen
✅ Automatic result calculation
✅ Clean and modular Python structure

🛠️ Technologies Used
Python 3.x
OpenCV
MediaPipe
NumPy


📂 Project Structure
Hand-Gesture-Math-Solver/
│
├── main.py
├── hand_tracking_module.py
├── requirements.txt
└── README.md


⚙️ Installation

1️⃣ Clone the repository:

git clone https://github.com/your-username/Hand-Gesture-Math-Solver.git
cd Hand-Gesture-Math-Solver

2️⃣ Install required libraries:

pip install -r requirements.txt

OR manually:

pip install mediapipe opencv-python numpy

▶️ How to Run
python main.py

Press Q to exit the webcam window.

🧠 How It Works
Webcam captures live video input.
MediaPipe detects 21 hand landmarks.
Finger positions are analyzed.
Each finger combination represents:
A number (0–5)
Or a mathematical operator
The expression is formed dynamically.
The program evaluates and displays the result on screen.


📊 Example

Gesture Input:
2 + 3
Output:
Result = 5


🎯 Applications
Gesture-based Human Computer Interaction
Educational tool for children
Touchless input systems
Computer Vision academic project

🔮 Future Improvements
Support for multi-digit numbers
Complex equations
GUI-based calculator interface
AI-based gesture classification
Accuracy optimization


👨‍💻 Author

Ayush Singh
B.Tech CSE Student
Python Developer

⭐ Support

If you found this project helpful, give it a ⭐ on GitHub!
BlindNav: AI-Powered Navigation for the Visually Impaired (PS-145)
BlindNav is an innovative Android-based "Artificial Eye" designed to empower visually impaired individuals. By leveraging Edge-AI, the application provides real-time environment scanning, currency identification, and emergency safety features through multi-modal feedback (Voice + Haptics).

** Key Features**
Real-Time Object Detection: Identifies obstacles like chairs, people, and vehicles using a quantized YOLO model for low-latency performance.

Currency Identification: Specialized mode for recognizing Indian Currency notes to foster financial independence.

Emergency SOS: Instant GPS location sharing via SMS to trusted guardians through voice commands or hardware triggers.

Multi-Language Support: Translates visual data into regional languages (Telugu, Hindi, etc.) using Google ML Kit.

**Intelligent Interaction**

Auto-Flashlight: Automatically activates in low-light environments to maintain AI accuracy.

Haptic Proximity: Vibrational alerts that intensify as objects get closer to the user.

** Technology Stack**
Frontend: Android (Kotlin / Jetpack Compose)

AI Engine: TensorFlow Lite (TFLite)

Computer Vision: CameraX API

Language Services: Google ML Kit (Translation & Text-to-Speech)

Location: Google Fused Location Provider API

** Project Architecture**
The app follows a modular architecture for high performance:

Image Analysis Layer: Captures frames via CameraX and preprocesses them for the AI model.

Inference Layer: Processes frames through a quantized YOLO11 model to detect objects and currency.

Feedback Layer: Converts detections into voice output (TTS) and haptic vibrations.

Safety Layer: Monitors triggers for SOS alerts and captures real-time GPS coordinates.

** The Team **
Hansitha: Core AI & TFLite Model Optimization

Madhu Sree: CameraX Integration & Frame Processing

Lasya: Logic Development (SOS & Haptics)

Supriya: Audio Interface & TTS Integration

Kavya: UI/UX Design & Documentation Lead

** Installation & Setup **
Clone the Repo:

Bash
git clone https://github.com/yourusername/BlindNav.git
Open in Android Studio: Ensure you have the latest version of Ladybug or higher.

Dependencies: Sync Gradle to download TensorFlow Lite, ML Kit, and CameraX libraries.

Permissions: Grant Camera, Location, and SMS permissions on the first run.

Run: Deploy on an Android device (API 24 or higher).

** Future Roadmap **
Integration with wearable Smart Glasses.

Offline regional language support.

Indoor floor-mapping for malls and university campuses.

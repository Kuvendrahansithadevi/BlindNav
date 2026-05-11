BlindNav is an innovative Android-based "Artificial Eye" designed to empower visually impaired individuals. By leveraging Edge-AI and a hybrid connectivity model, the application provides real-time environment scanning, currency identification, and emergency safety features through multi-modal feedback.

**Key Features**
Real-Time Object Detection: Identifies 80+ categories like chairs, people, and vehicles using an optimized YOLO11n TFLite model for low-latency performance.

Currency Identification: Specialized mode for recognizing Indian Currency notes (₹10 to ₹500) to foster financial independence.

Dynamic Variable Haptics: Unlike static alerts, the vibration intensity and frequency scale proportionally based on object proximity, providing a tactile "sense of distance."

Multilingual Voice Interface: Provides auditory feedback in English, Telugu, and Hindi using Google ML Kit to bridge the linguistic divide.

Offline-Resilient SOS: A critical safety feature that works without internet. Users can trigger an instant GPS-linked SMS to trusted guardians via hardware button gestures.

*Technology Stack*
Frontend: Kotlin / Jetpack Compose (Modern & Responsive UI)

AI Engine: TensorFlow Lite (TFLite) with INT8 Quantization

Computer Vision: CameraX API for real-time frame analysis

Language Services: Google ML Kit (Translation & Text-to-Speech)

Connectivity: Hybrid (Online for AI Inference, 100% Offline for Emergency SOS)

*Project Architecture*
The app follows a modular architecture to ensure performance and reliability:

Vision Pipeline: Captures and preprocesses frames via CameraX.

Inference Layer: Processes frames locally/cloud via YOLO models to detect objects/currency.

Proximity Engine: Calculates bounding box scale to trigger Variable Haptic Intensity.

Safety Layer: Monitors hardware triggers for Offline SMS telemetry.

*The Team*
Hansitha: Core AI & TFLite Model Optimization

Supriya: Audio Interface, TTS Integration & Logic Refinement

Madhu Sree: CameraX Integration & Frame Processing

Kavya: Documentation Lead & Emergency SOS

Lasya:BlindNav is an innovative Android-based "Artificial Eye" designed to empower visually impaired individuals. By leveraging Edge-AI and a hybrid connectivity model, the application provides real-time environment scanning, currency identification, and emergency safety features through multi-modal feedback.

**Key Features**
Real-Time Object Detection: Identifies 80+ categories like chairs, people, and vehicles using an optimized YOLO11n TFLite model for low-latency performance.

Currency Identification: Specialized mode for recognizing Indian Currency notes (₹10 to ₹500) to foster financial independence.

Dynamic Variable Haptics: Unlike static alerts, the vibration intensity and frequency scale proportionally based on object proximity, providing a tactile "sense of distance."

Multilingual Voice Interface: Provides auditory feedback in English, Telugu, and Hindi using Google ML Kit to bridge the linguistic divide.

Offline-Resilient SOS: A critical safety feature that works without internet. Users can trigger an instant GPS-linked SMS to trusted guardians via hardware button gestures.

*Technology Stack*
Frontend: Kotlin / Jetpack Compose (Modern & Responsive UI)

AI Engine: TensorFlow Lite (TFLite) with INT8 Quantization

Computer Vision: CameraX API for real-time frame analysis

Language Services: Google ML Kit (Translation & Text-to-Speech)

Connectivity: Hybrid (Online for AI Inference, 100% Offline for Emergency SOS)

*Project Architecture*
The app follows a modular architecture to ensure performance and reliability:

Vision Pipeline: Captures and preprocesses frames via CameraX.

Inference Layer: Processes frames locally/cloud via YOLO models to detect objects/currency.

Proximity Engine: Calculates bounding box scale to trigger Variable Haptic Intensity.

Safety Layer: Monitors hardware triggers for Offline SMS telemetry.

*The Team*
Hansitha: Core AI & TFLite Model Optimization

Supriya: Audio Interface, TTS Integration & Logic Refinement

Madhu Sree: CameraX Integration & Frame Processing

Kavya: UI/UX Design & Documentation & Emergency SOS

Lasya: Haptic Feedback Integration

 Future Roadmap
On-Device OCR: Integrated content reader for books and signboards.

Indoor Mapping: Floor-mapping for malls and university campuses.

Wearable Integration: Support for smart-glass hardware. Haptic Feedback Integration

 Future Roadmap
On-Device OCR: Integrated content reader for books and signboards.

Indoor Mapping: Floor-mapping for malls and university campuses.

Wearable Integration: Support for smart-glass hardware.

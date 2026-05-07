package com.example.ainavigationforblindpeople

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.ViewGroup
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import java.util.Locale
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private var objectDetector: ObjectDetector? = null
    private var tts: TextToSpeech? = null
    private var lastSpokenTime = 0L
    private val speechInterval = 3000L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setupObjectDetector()
        setupTTS()

        setContent {
            var hasCameraPermission by remember {
                mutableStateOf(
                    ContextCompat.checkSelfPermission(
                        this,
                        Manifest.permission.CAMERA
                    ) == PackageManager.PERMISSION_GRANTED
                )
            }

            val launcher = rememberLauncherForActivityResult(
                contract = ActivityResultContracts.RequestPermission(),
                onResult = { granted ->
                    hasCameraPermission = granted
                }
            )

            LaunchedEffect(Unit) {
                if (!hasCameraPermission) {
                    launcher.launch(Manifest.permission.CAMERA)
                }
            }

            if (hasCameraPermission) {
                ObjectDetectionScreen()
            }
        }
    }

    private fun setupObjectDetector() {
        val baseOptionsBuilder = BaseOptions.builder()
        
        // --- Hardware Acceleration (GPU) ---
        val compatibilityList = CompatibilityList()
        if (compatibilityList.isDelegateSupportedOnThisDevice) {
            baseOptionsBuilder.useGpu()
            Log.d("TFLite", "Using GPU Delegate")
        } else {
            baseOptionsBuilder.setNumThreads(4)
            Log.d("TFLite", "Using CPU (4 threads)")
        }

        // --- Confidence Threshold (0.55) ---
        val optionsBuilder = ObjectDetector.ObjectDetectorOptions.builder()
            .setScoreThreshold(0.55f) // Set to 55% as requested
            .setMaxResults(3)        // Limit results to reduce noise
            .setBaseOptions(baseOptionsBuilder.build())

        try {
            objectDetector = ObjectDetector.createFromFileAndOptions(
                this, 
                "detect.tflite", 
                optionsBuilder.build()
            )
            Log.d("TFLite", "ObjectDetector initialized successfully")
        } catch (e: Exception) {
            Log.e("TFLite", "Error initializing ObjectDetector", e)
        }
    }

    private fun setupTTS() {
        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale.US
                speak("App ready.")
            }
        }
    }

    @Composable
    fun ObjectDetectionScreen() {
        val lifecycleOwner = LocalLifecycleOwner.current
        val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

        Box(modifier = Modifier.fillMaxSize()) {
            AndroidView(
                modifier = Modifier.fillMaxSize(),
                factory = { ctx ->
                    val previewView = PreviewView(ctx).apply {
                        layoutParams = ViewGroup.LayoutParams(
                            ViewGroup.LayoutParams.MATCH_PARENT,
                            ViewGroup.LayoutParams.MATCH_PARENT
                        )
                    }

                    val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                    cameraProviderFuture.addListener({
                        val cameraProvider = cameraProviderFuture.get()

                        val preview = Preview.Builder().build().also {
                            it.setSurfaceProvider(previewView.surfaceProvider)
                        }

                        val imageAnalysis = ImageAnalysis.Builder()
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                            .build()

                        imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                            detectObjects(imageProxy)
                        }

                        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                        try {
                            cameraProvider.unbindAll()
                            cameraProvider.bindToLifecycle(
                                lifecycleOwner,
                                cameraSelector,
                                preview,
                                imageAnalysis
                            )
                        } catch (e: Exception) {
                            Log.e("CameraX", "Use case binding failed", e)
                        }
                    }, ContextCompat.getMainExecutor(ctx))

                    previewView
                }
            )
        }
    }

    private fun detectObjects(imageProxy: ImageProxy) {
        val detector = objectDetector ?: return

        // 1. Convert ImageProxy to Bitmap
        val bitmap = imageProxy.toBitmap()
        
        // 2. Preprocessing: Handle rotation and resize without stretching
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val size = minOf(bitmap.width, bitmap.height)
        
        val imageProcessor = ImageProcessor.Builder()
            .add(Rot90Op(-rotationDegrees / 90))
            .add(ResizeWithCropOrPadOp(size, size)) // Center crop to square
            .add(ResizeOp(300, 300, ResizeOp.Method.BILINEAR)) // Scale to 300x300
            .build()

        // 3. Convert Bitmap to TensorImage and process
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))

        try {
            // 4. Run inference
            val results = detector.detect(tensorImage)

            // 5. Announce results
            val bestDetection = results.firstOrNull()
            val category = bestDetection?.categories?.firstOrNull()
            
            if (category != null && category.score >= 0.55f) {
                val label = category.label
                val currentTime = System.currentTimeMillis()
                if (currentTime - lastSpokenTime > speechInterval) {
                    speak("I see a $label")
                    lastSpokenTime = currentTime
                }
            }
        } catch (e: Exception) {
            Log.e("TFLite", "Inference failed", e)
        } finally {
            imageProxy.close()
        }
    }

    private fun speak(text: String) {
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
    }

    override fun onDestroy() {
        super.onDestroy()
        tts?.shutdown()
        objectDetector?.close()
    }
}

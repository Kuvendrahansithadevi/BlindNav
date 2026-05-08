package com.example.ainavigationforblindpeople

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Paint
import android.graphics.RectF
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.util.Locale
import java.util.concurrent.Executors

data class DetectionResult(val label: String, val score: Float, val box: RectF)

class MainActivity : ComponentActivity(), SensorEventListener {
    private var interpreter: Interpreter? = null
    private var labels = listOf<String>()
    private var tts: TextToSpeech? = null
    private var detectionsState by mutableStateOf<List<DetectionResult>>(emptyList())
    private var lastSpokenTime = 0L
    private val speechInterval = 3000L

    private lateinit var sensorManager: SensorManager
    private var lightSensor: Sensor? = null
    private var isEnvironmentDark by mutableStateOf(false)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setupInterpreter()
        setupTTS()
        setupLightSensor()
        setContent {
            var hasPermission by remember {
                mutableStateOf(ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
            }
            val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { hasPermission = it }

            LaunchedEffect(Unit) {
                if (!hasPermission) launcher.launch(Manifest.permission.CAMERA)
            }

            if (hasPermission) {
                ObjectDetectionScreen(detectionsState, isEnvironmentDark) { imageProxy -> detectObjects(imageProxy) }
            }
        }
    }

    private fun setupLightSensor() {
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        lightSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LIGHT)
    }

    override fun onResume() {
        super.onResume()
        lightSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_UI)
        }
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event?.sensor?.type == Sensor.TYPE_LIGHT) {
            val lux = event.values[0]
            // Threshold for dark environment: < 15 lux is considered dark
            isEnvironmentDark = lux < 15f
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun setupInterpreter() {
        try {
            val model = FileUtil.loadMappedFile(this, "yolo11n_int8.tflite")
            val options = Interpreter.Options().setNumThreads(4)
            interpreter = Interpreter(model, options)

            // Log model details to verify fix
            val inputTensor = interpreter!!.getInputTensor(0)
            Log.d("BlindNav", "Model Loaded. Expected Input: ${inputTensor.dataType()} ${inputTensor.shape().contentToString()}")

            labels = loadLabels()
        } catch (e: Exception) {
            Log.e("BlindNav", "Model Load Error: ${e.message}")
        }
    }

    private fun detectObjects(imageProxy: ImageProxy) {
        val interp = interpreter ?: return
        val bitmap = imageProxy.toBitmap()

        // FIX: Re-added NormalizeOp because Logcat shows model expects 4.9MB (FLOAT32)
        val imageProcessor = ImageProcessor.Builder()
            .add(Rot90Op(-imageProxy.imageInfo.rotationDegrees / 90))
            .add(ResizeOp(640, 640, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .build()

        // FIX: Changed DataType to FLOAT32 to match 4915200 bytes requirement
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        val processedImage = imageProcessor.process(tensorImage)

        val output = Array(1) { Array(84) { FloatArray(8400) } }

        try {
            interp.run(processedImage.buffer, output)
            val data = output[0]
            val results = mutableListOf<DetectionResult>()

            for (i in 0 until 8400) {
                var maxScore = 0f
                var classId = -1
                for (c in 0 until 80) {
                    val score = data[c + 4][i]
                    if (score > maxScore) {
                        maxScore = score
                        classId = c
                    }
                }

                // Increased threshold slightly for cleaner results
                if (maxScore > 0.40f) {
                    val cx = data[0][i]
                    val cy = data[1][i]
                    val w = data[2][i]
                    val h = data[3][i]

                    results.add(DetectionResult(
                        labels.getOrElse(classId) { "Object" },
                        maxScore,
                        RectF((cx - w / 2f) / 640f, (cy - h / 2f) / 640f, (cx + w / 2f) / 640f, (cy + h / 2f) / 640f)
                    ))
                }
            }

            val finalResults = applyNMS(results)

            runOnUiThread {
                detectionsState = finalResults
                if (finalResults.isNotEmpty()) {
                    val top = finalResults.maxBy { it.score }
                    val currentTime = System.currentTimeMillis()
                    if (currentTime - lastSpokenTime > speechInterval) {
                        tts?.speak(top.label, TextToSpeech.QUEUE_FLUSH, null, null)
                        lastSpokenTime = currentTime
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("BlindNav", "Inference Error: ${e.message}")
        } finally {
            imageProxy.close()
        }
    }

    private fun loadLabels() = listOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
        // Indian Currency labels (Requires a specialized model)
        "10 Rupee", "20 Rupee", "50 Rupee", "100 Rupee", "200 Rupee", "500 Rupee"
    )

    private fun setupTTS() {
        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale.US
                tts?.speak("AI Navigation ready", TextToSpeech.QUEUE_FLUSH, null, null)
            }
        }
    }

    private fun applyNMS(results: List<DetectionResult>): List<DetectionResult> {
        val sortedResults = results.sortedByDescending { it.score }
        val selectedResults = mutableListOf<DetectionResult>()

        for (result in sortedResults) {
            var keep = true
            for (selected in selectedResults) {
                if (calculateIoU(result.box, selected.box) > 0.45f) {
                    keep = false
                    break
                }
            }
            if (keep) {
                selectedResults.add(result)
                if (selectedResults.size >= 5) break
            }
        }
        return selectedResults
    }

    private fun calculateIoU(box1: RectF, box2: RectF): Float {
        val intL = maxOf(box1.left, box2.left)
        val intT = maxOf(box1.top, box2.top)
        val intR = minOf(box1.right, box2.right)
        val intB = minOf(box1.bottom, box2.bottom)

        val intW = maxOf(0f, intR - intL)
        val intH = maxOf(0f, intB - intT)
        val intArea = intW * intH

        val area1 = (box1.right - box1.left) * (box1.bottom - box1.top)
        val area2 = (box2.right - box2.left) * (box2.bottom - box2.top)
        val unionArea = area1 + area2 - intArea

        return if (unionArea > 0) intArea / unionArea else 0f
    }

    override fun onDestroy() {
        super.onDestroy()
        tts?.shutdown()
        interpreter?.close()
    }
}

@Composable
fun ObjectDetectionScreen(
    detections: List<DetectionResult>,
    isDark: Boolean,
    onAnalyze: (ImageProxy) -> Unit
) {
    val lifecycleOwner = LocalLifecycleOwner.current
    val executor = remember { Executors.newSingleThreadExecutor() }
    var camera by remember { mutableStateOf<Camera?>(null) }

    // Toggle torch based on ambient light
    LaunchedEffect(isDark) {
        camera?.cameraControl?.enableTorch(isDark)
    }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                PreviewView(ctx).apply {
                    implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                    val providerFuture = ProcessCameraProvider.getInstance(ctx)
                    providerFuture.addListener({
                        val provider = providerFuture.get()
                        val preview = Preview.Builder().build().also {
                            it.setSurfaceProvider(surfaceProvider)
                        }
                        val analysis = ImageAnalysis.Builder()
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                            .build().also {
                                it.setAnalyzer(executor) { proxy -> onAnalyze(proxy) }
                            }
                        provider.unbindAll()
                        camera = provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
                    }, ContextCompat.getMainExecutor(ctx))
                }
            }
        )

        Canvas(modifier = Modifier.fillMaxSize()) {
            val paint = Paint().apply {
                color = android.graphics.Color.GREEN
                style = Paint.Style.STROKE
                strokeWidth = 8f
            }
            val textPaint = Paint().apply {
                color = android.graphics.Color.WHITE
                textSize = 40f
            }

            drawIntoCanvas { canvas ->
                detections.forEach { detection ->
                    val rect = RectF(
                        detection.box.left * size.width,
                        detection.box.top * size.height,
                        detection.box.right * size.width,
                        detection.box.bottom * size.height
                    )
                    canvas.nativeCanvas.drawRect(rect, paint)
                    canvas.nativeCanvas.drawText(detection.label, rect.left, rect.top - 10f, textPaint)
                }
            }
        }
    }
}
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
import android.content.Intent
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.example.ainavigationforblindpeople.ui.theme.AINavigationForBlindPeopleTheme
import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.common.model.RemoteModelManager
import com.google.mlkit.nl.translate.TranslateLanguage
import com.google.mlkit.nl.translate.TranslateRemoteModel
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.Translator
import com.google.mlkit.nl.translate.TranslatorOptions
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.util.*
import java.util.concurrent.Executors

data class DetectionResult(val label: String, val score: Float, val box: RectF)

enum class AppMode { OBJECT_DETECTION, CURRENCY_DETECTION }

class MainActivity : ComponentActivity(), SensorEventListener {
    private var interpreter: Interpreter? = null
    private var labels = listOf<String>()
    private var tts: TextToSpeech? = null
    private var detectionsState by mutableStateOf<List<DetectionResult>>(emptyList())
    private var lastSpokenTime = 0L
    private val speechInterval = 4000L // 4 seconds gap for translation + speaking

    // Translation State
    private var selectedLangCode by mutableStateOf(TranslateLanguage.ENGLISH)
    private var selectedLangName by mutableStateOf("English")
    private var isDownloadingModel by mutableStateOf(false)
    private var translator: Translator? = null
    private var currentTranslatorLang: String? = null

    private lateinit var sensorManager: SensorManager
    private var lightSensor: Sensor? = null
    private var isEnvironmentDark by mutableStateOf(false)

    private var numClasses = 0
    private var numRows = 0
    private var inputSize = 640

    private var isDetectionActive by mutableStateOf(false)
    private var speechRecognizer: SpeechRecognizer? = null
    private lateinit var speechIntent: Intent
    private var isListening by mutableStateOf(false)
    private var currentMode by mutableStateOf(AppMode.OBJECT_DETECTION)

    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setupInterpreter()
        setupTTS()
        setupLightSensor()
        setupSpeechRecognizer()

        setContent {
            CompositionLocalProvider(LocalLifecycleOwner provides this) {
                AINavigationForBlindPeopleTheme {
                    val permissions = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
                    var permissionsGranted by remember {
                        mutableStateOf(permissions.all {
                            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
                        })
                    }
                    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { results ->
                        permissionsGranted = results.values.all { it }
                        if (permissionsGranted) {
                            speechRecognizer?.startListening(speechIntent)
                        }
                    }

                    LaunchedEffect(Unit) {
                        if (!permissionsGranted) {
                            launcher.launch(permissions)
                        } else {
                            speechRecognizer?.startListening(speechIntent)
                        }
                    }

                    LaunchedEffect(selectedLangCode) {
                        ensureModelDownloaded(selectedLangCode)
                    }

                    if (permissionsGranted) {
                        Scaffold(
                            topBar = {
                                TopAppBar(
                                    title = { Text("AI Navigator") },
                                    actions = {
                                        var showMenu by remember { mutableStateOf(false) }
                                        IconButton(onClick = { showMenu = true }) {
                                            Icon(Icons.Default.Settings, contentDescription = "Settings")
                                        }
                                        DropdownMenu(
                                            expanded = showMenu,
                                            onDismissRequest = { showMenu = false }
                                        ) {
                                            val languages = listOf(
                                                "English" to TranslateLanguage.ENGLISH,
                                                "Telugu" to TranslateLanguage.TELUGU,
                                                "Hindi" to TranslateLanguage.HINDI,
                                                "Tamil" to TranslateLanguage.TAMIL,
                                                "Kannada" to TranslateLanguage.KANNADA
                                            )
                                            languages.forEach { (name, code) ->
                                                DropdownMenuItem(
                                                    text = { Text(name) },
                                                    onClick = {
                                                        selectedLangName = name
                                                        selectedLangCode = code
                                                        showMenu = false
                                                        speakFeedback("Language changed to $name")
                                                        ensureModelDownloaded(code)
                                                    }
                                                )
                                            }
                                        }
                                    },
                                    colors = TopAppBarDefaults.topAppBarColors(
                                        containerColor = MaterialTheme.colorScheme.primaryContainer,
                                        titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer
                                    )
                                )
                            }
                        ) { paddingValues ->
                            Box(modifier = Modifier.fillMaxSize().padding(paddingValues)) {
                                // 1. Camera Layer (Hansitha's Logic) - Full Screen
                                ObjectDetectionScreen(detectionsState, isEnvironmentDark) { imageProxy ->
                                    detectObjects(imageProxy)
                                }

                                // 2. Voice Status Overlay (Kept for accessibility confirmation)
                                VoiceStatusOverlay(isListening, isDetectionActive, currentMode)
                            }
                        }
                    }
                }
            }
        }
    }

    // --- SPEECH RECOGNITION LOGIC ---
    private fun setupSpeechRecognizer() {
        if (!SpeechRecognizer.isRecognitionAvailable(this)) return
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        speechIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.US.toString())
        }

        speechRecognizer?.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) { isListening = true }
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() { isListening = false }
            override fun onError(error: Int) {
                isListening = false
                speechRecognizer?.startListening(speechIntent)
            }
            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                matches?.get(0)?.let { handleVoiceCommand(it.lowercase()) }
                speechRecognizer?.startListening(speechIntent)
            }
            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
    }

    private fun handleVoiceCommand(command: String) {
        when {
            command.contains("check money") || command.contains("currency") -> {
                currentMode = AppMode.CURRENCY_DETECTION
                speakFeedback("Currency Detection mode activated. Please hold the note in front of the camera.")
            }
            command.contains("detect objects") || command.contains("back") || command.contains("object mode") -> {
                currentMode = AppMode.OBJECT_DETECTION
                speakFeedback("Object Detection mode activated.")
            }
            command.contains("start") -> {
                isDetectionActive = true
                speakFeedback("Starting Object Detection")
            }
            command.contains("stop") -> {
                isDetectionActive = false
                speakFeedback("Stopping Object Detection")
                detectionsState = emptyList()
            }
            command.contains("english") -> {
                selectedLangName = "English"
                selectedLangCode = TranslateLanguage.ENGLISH
                speakFeedback("Language changed to English")
                isDownloadingModel = false
            }
            command.contains("telugu") -> {
                selectedLangName = "Telugu"
                selectedLangCode = TranslateLanguage.TELUGU
                speakFeedback("Language changed to Telugu")
                ensureModelDownloaded(TranslateLanguage.TELUGU)
            }
            command.contains("hindi") -> {
                selectedLangName = "Hindi"
                selectedLangCode = TranslateLanguage.HINDI
                speakFeedback("Language changed to Hindi")
                ensureModelDownloaded(TranslateLanguage.HINDI)
            }
            command.contains("tamil") -> {
                selectedLangName = "Tamil"
                selectedLangCode = TranslateLanguage.TAMIL
                speakFeedback("Language changed to Tamil")
                ensureModelDownloaded(TranslateLanguage.TAMIL)
            }
            command.contains("kannada") -> {
                selectedLangName = "Kannada"
                selectedLangCode = TranslateLanguage.KANNADA
                speakFeedback("Language changed to Kannada")
                ensureModelDownloaded(TranslateLanguage.KANNADA)
            }
        }
    }

    private fun speakFeedback(text: String) {
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
    }

    // --- TRANSLATION LOGIC (Merged) ---
    private fun ensureModelDownloaded(langCode: String) {
        if (langCode == TranslateLanguage.ENGLISH) {
            isDownloadingModel = false
            return
        }

        // Initialize translator for this language if needed
        if (currentTranslatorLang != langCode) {
            translator?.close()
            val options = TranslatorOptions.Builder()
                .setSourceLanguage(TranslateLanguage.ENGLISH)
                .setTargetLanguage(langCode)
                .build()
            translator = Translation.getClient(options)
            currentTranslatorLang = langCode
        }

        val modelManager = RemoteModelManager.getInstance()
        val model = TranslateRemoteModel.Builder(langCode).build()

        modelManager.isModelDownloaded(model).addOnSuccessListener { isDownloaded ->
            if (!isDownloaded) {
                isDownloadingModel = true
                speakFeedback("Downloading translation model for $selectedLangName, please wait.")
                val conditions = DownloadConditions.Builder().build()
                translator?.downloadModelIfNeeded(conditions)
                    ?.addOnSuccessListener {
                        isDownloadingModel = false
                        speakFeedback("$selectedLangName model ready.")
                    }
                    ?.addOnFailureListener {
                        isDownloadingModel = false
                        speakFeedback("Failed to download $selectedLangName model. Detection will continue in English.")
                    }
            } else {
                isDownloadingModel = false
            }
        }
    }

    private fun translateAndSpeak(text: String) {
        // If it's a currency detection, speak directly for clarity
        if (text.contains("Rupee")) {
            val clearText = text.replace("Rupee", "Rupees")
            tts?.speak("Detected $clearText", TextToSpeech.QUEUE_FLUSH, null, null)
            return
        }

        if (selectedLangCode == TranslateLanguage.ENGLISH) {
            tts?.language = Locale.US
            tts?.speak("There is a $text", TextToSpeech.QUEUE_FLUSH, null, null)
            return
        }

        if (isDownloadingModel) {
            tts?.speak("Translation model downloading, please wait", TextToSpeech.QUEUE_FLUSH, null, null)
            return
        }

        // Ensure translator is ready for the current language
        if (currentTranslatorLang != selectedLangCode || translator == null) {
            ensureModelDownloaded(selectedLangCode)
            // Fallback to English for this frame while we re-initialize/download
            tts?.language = Locale.US
            tts?.speak("There is a $text", TextToSpeech.QUEUE_FLUSH, null, null)
            return
        }

        translator?.translate("There is a $text ahead")
            ?.addOnSuccessListener { translatedText ->
                tts?.language = Locale(selectedLangCode)
                tts?.speak(translatedText, TextToSpeech.QUEUE_FLUSH, null, null)
            }
            ?.addOnFailureListener { e ->
                Log.e("BlindNav", "Translation failed: ${e.message}")
                // If translation fails, trigger download check and fallback to English
                ensureModelDownloaded(selectedLangCode)
                tts?.language = Locale.US
                tts?.speak("There is a $text", TextToSpeech.QUEUE_FLUSH, null, null)
            }
    }

    private fun detectObjects(imageProxy: ImageProxy) {
        try {
            if (!isDetectionActive) {
                return
            }
            val interp = interpreter ?: return
            val bitmap = imageProxy.toBitmap() ?: return

            val imageProcessor = ImageProcessor.Builder()
                .add(Rot90Op(-imageProxy.imageInfo.rotationDegrees / 90))
                .add(ResizeOp(inputSize, inputSize, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f))
                .build()

            val tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            val processedImage = imageProcessor.process(tensorImage)

            val output = Array(1) { Array(numRows) { FloatArray(8400) } }

            interp.run(processedImage.buffer, output)
            val data = output[0]
            val results = mutableListOf<DetectionResult>()

            for (i in 0 until 8400) {
                var maxScore = 0f
                var classId = -1
                for (c in 0 until numClasses) {
                    val score = data[c + 4][i]
                    if (score > maxScore) {
                        maxScore = score
                        classId = c
                    }
                }

                if (maxScore > 0.50f) { // Balanced threshold for better accuracy
                    val cx = data[0][i]
                    val cy = data[1][i]
                    val w = data[2][i]
                    val h = data[3][i]

                    results.add(DetectionResult(
                        labels.getOrElse(classId) { "Object" },
                        maxScore,
                        RectF((cx - w / 2f) / inputSize.toFloat(), (cy - h / 2f) / inputSize.toFloat(), (cx + w / 2f) / inputSize.toFloat(), (cy + h / 2f) / inputSize.toFloat())
                    ))
                }
            }

            val finalResults = applyNMS(results)

            // Filter results based on current mode
            val filteredResults = when (currentMode) {
                AppMode.CURRENCY_DETECTION -> finalResults.filter { it.label.contains("Rupee") }
                AppMode.OBJECT_DETECTION -> finalResults.filter { !it.label.contains("Rupee") }
            }

            runOnUiThread {
                detectionsState = filteredResults
                if (filteredResults.isNotEmpty()) {
                    val top = filteredResults.maxBy { it.score }
                    val currentTime = System.currentTimeMillis()
                    if (currentTime - lastSpokenTime > speechInterval) {
                        // MERGED: Call translation instead of direct TTS
                        translateAndSpeak(top.label)
                        lastSpokenTime = currentTime
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("BlindNav", "Detection Error: ${e.message}")
        } finally {
            imageProxy.close()
        }
    }

    // --- ALL SUPPORT FUNCTIONS FROM HANSITHA'S CODE (RETAINED) ---
    private fun setupLightSensor() {
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        lightSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LIGHT)
    }

    override fun onResume() {
        super.onResume()
        lightSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_UI) }
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event?.sensor?.type == Sensor.TYPE_LIGHT) {
            isEnvironmentDark = event.values[0] < 15f
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun setupInterpreter() {
        try {
            val model = FileUtil.loadMappedFile(this, "yolo11n_int8.tflite")
            val options = Interpreter.Options().setNumThreads(4)
            interpreter = Interpreter(model, options)
            labels = loadLabels()

            // Dynamically initialize model properties
            val inputShape = interpreter?.getInputTensor(0)?.shape()
            inputSize = inputShape?.get(1) ?: 640

            val outputShape = interpreter?.getOutputTensor(0)?.shape()
            numRows = outputShape?.get(1) ?: 84
            numClasses = numRows - 4

            Log.d("BlindNav", "Model initialized: InputSize=$inputSize, Classes=$numClasses")
        } catch (e: Exception) { Log.e("BlindNav", "Model Load Error: ${e.message}") }
    }

    private fun setupTTS() {
        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale.US
                tts?.setSpeechRate(0.8f)
            }
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
        "10 Rupee", "20 Rupee", "50 Rupee", "100 Rupee", "200 Rupee", "500 Rupee"
    )

    private fun applyNMS(results: List<DetectionResult>): List<DetectionResult> {
        val sortedResults = results.sortedByDescending { it.score }
        val selectedResults = mutableListOf<DetectionResult>()
        for (result in sortedResults) {
            var keep = true
            for (selected in selectedResults) {
                if (calculateIoU(result.box, selected.box) > 0.45f) { keep = false; break }
            }
            if (keep) {
                selectedResults.add(result)
                if (selectedResults.size >= 5) break
            }
        }
        return selectedResults
    }

    private fun calculateIoU(box1: RectF, box2: RectF): Float {
        val intL = maxOf(box1.left, box2.left); val intT = maxOf(box1.top, box2.top)
        val intR = minOf(box1.right, box2.right); val intB = minOf(box1.bottom, box2.bottom)
        val intArea = maxOf(0f, intR - intL) * maxOf(0f, intB - intT)
        val area1 = (box1.right - box1.left) * (box1.bottom - box1.top)
        val area2 = (box2.right - box2.left) * (box2.bottom - box2.top)
        return if (area1 + area2 - intArea > 0) intArea / (area1 + area2 - intArea) else 0f
    }

    override fun onDestroy() {
        super.onDestroy()
        tts?.shutdown(); interpreter?.close()
        speechRecognizer?.destroy()
        translator?.close()
    }
}

// --- UI COMPONENTS ---

@Composable
fun VoiceStatusOverlay(isListening: Boolean, isDetectionActive: Boolean, currentMode: AppMode) {
    Box(modifier = Modifier.fillMaxSize().padding(bottom = 20.dp), contentAlignment = Alignment.BottomCenter) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.background(Color.Black.copy(alpha = 0.5f), shape = MaterialTheme.shapes.small).padding(8.dp)
        ) {
            if (isListening) {
                Text("Listening...", color = Color.Cyan, style = MaterialTheme.typography.labelMedium)
            }
            Text(
                "MODE: ${currentMode.name.replace("_", " ")}",
                color = Color.White,
                style = MaterialTheme.typography.labelSmall
            )
            Text(
                if (isDetectionActive) "DETECTION: ON" else "DETECTION: OFF",
                color = if (isDetectionActive) Color.Green else Color.Red,
                style = MaterialTheme.typography.titleSmall
            )
        }
    }
}

@Composable
fun ObjectDetectionScreen(detections: List<DetectionResult>, isDark: Boolean, onAnalyze: (ImageProxy) -> Unit) {
    val lifecycleOwner = LocalLifecycleOwner.current
    val executor = remember { Executors.newSingleThreadExecutor() }
    var camera by remember { mutableStateOf<Camera?>(null) }

    LaunchedEffect(isDark) { camera?.cameraControl?.enableTorch(isDark) }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                PreviewView(ctx).apply {
                    val providerFuture = ProcessCameraProvider.getInstance(ctx)
                    providerFuture.addListener({
                        val provider = providerFuture.get()
                        val preview = Preview.Builder().build().also { it.setSurfaceProvider(surfaceProvider) }
                        val analysis = ImageAnalysis.Builder()
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                            .build().also { it.setAnalyzer(executor) { proxy -> onAnalyze(proxy) } }
                        provider.unbindAll()
                        camera = provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
                    }, ContextCompat.getMainExecutor(ctx))
                }
            }
        )

        Canvas(modifier = Modifier.fillMaxSize()) {
            val paint = Paint().apply { color = android.graphics.Color.GREEN; style = Paint.Style.STROKE; strokeWidth = 8f }
            val textPaint = Paint().apply { color = android.graphics.Color.WHITE; textSize = 40f }
            drawIntoCanvas { canvas ->
                detections.forEach { detection ->
                    val rect = RectF(detection.box.left * this.size.width, detection.box.top * this.size.height, detection.box.right * this.size.width, detection.box.bottom * this.size.height)
                    canvas.nativeCanvas.drawRect(rect, paint)
                    canvas.nativeCanvas.drawText(detection.label, rect.left, rect.top - 10f, textPaint)
                }
            }
        }
    }
}
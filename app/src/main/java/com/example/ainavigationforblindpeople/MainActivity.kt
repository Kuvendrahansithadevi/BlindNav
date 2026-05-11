package com.example.ainavigationforblindpeople

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Paint
import android.graphics.RectF
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.telephony.SmsManager
import android.util.Log
import android.view.KeyEvent
import android.widget.Toast
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
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.common.model.RemoteModelManager
import com.google.mlkit.nl.translate.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import java.util.*
import java.util.concurrent.Executors

// --- Data Classes & Enums ---
data class DetectionResult(val label: String, val score: Float, val box: RectF)
enum class AppMode { OBJECT_DETECTION, CURRENCY_DETECTION }

class MainActivity : ComponentActivity(), SensorEventListener {
    private var interpreter: Interpreter? = null
    private var currencyInterpreter: Interpreter? = null
    private var labels = listOf<String>()
    private var currencyLabels = listOf<String>()
    private var tts: TextToSpeech? = null
    private var detectionsState by mutableStateOf<List<DetectionResult>>(emptyList())
    private var lastSpokenTime = 0L
    private val speechInterval = 4000L

    // SOS Variables
    private var pressStartTime: Long = 0
    private val sosThreshold = 3000

    // Translation State
    private var selectedLangCode by mutableStateOf(TranslateLanguage.ENGLISH)
    private var selectedLangName by mutableStateOf("English")
    private var isDownloadingModel by mutableStateOf(false)
    private var translator: Translator? = null
    private var currentTranslatorLang: String? = null

    // Sensor & Detection State
    private lateinit var sensorManager: SensorManager
    private var lightSensor: Sensor? = null
    private var isEnvironmentDark by mutableStateOf(false)
    private var isDetectionActive by mutableStateOf(false)
    private var currentMode by mutableStateOf(AppMode.OBJECT_DETECTION)

    // Voice & Vibration
    private var speechRecognizer: SpeechRecognizer? = null
    private lateinit var speechIntent: Intent
    private var isListening by mutableStateOf(false)

    private val vibrator by lazy {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vibratorManager = getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
            vibratorManager.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setupInterpreter()
        setupTTS()
        setupLightSensor()
        setupSpeechRecognizer()

        setContent {
            CompositionLocalProvider(LocalLifecycleOwner provides this) {
                AINavigationForBlindPeopleTheme {
                    val requiredPermissions = arrayOf(
                        Manifest.permission.CAMERA,
                        Manifest.permission.RECORD_AUDIO,
                        Manifest.permission.SEND_SMS,
                        Manifest.permission.ACCESS_FINE_LOCATION,
                        Manifest.permission.VIBRATE
                    )

                    var permissionsGranted by remember {
                        mutableStateOf(requiredPermissions.all {
                            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
                        })
                    }

                    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { results ->
                        permissionsGranted = results.values.all { it }
                        if (permissionsGranted) {
                            checkAndEnableGPS(this)
                            speechRecognizer?.startListening(speechIntent)
                        }
                    }

                    LaunchedEffect(Unit) {
                        if (!permissionsGranted) launcher.launch(requiredPermissions)
                        else speechRecognizer?.startListening(speechIntent)
                    }

                    LaunchedEffect(selectedLangCode) { ensureModelDownloaded(selectedLangCode) }

                    if (permissionsGranted) {
                        Box(modifier = Modifier.fillMaxSize()) {
                            ObjectDetectionScreen(detectionsState, isEnvironmentDark) { imageProxy ->
                                detectObjects(imageProxy)
                            }
                            VoiceStatusOverlay(isListening, isDetectionActive, currentMode)
                        }
                    }
                }
            }
        }
    }

    // --- SOS VOLUME BUTTON LOGIC ---
    override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean {
        if (keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) {
            if (event.repeatCount == 0) pressStartTime = System.currentTimeMillis()
            else if (System.currentTimeMillis() - pressStartTime >= sosThreshold && pressStartTime != 0L) {
                triggerSOS()
                pressStartTime = 0L
                return true
            }
        }
        return super.onKeyDown(keyCode, event)
    }

    override fun onKeyUp(keyCode: Int, event: KeyEvent): Boolean {
        if (keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) pressStartTime = 0L
        return super.onKeyUp(keyCode, event)
    }

    @SuppressLint("MissingPermission")
    private fun triggerSOS() {
        triggerVibration(1000)
        val fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        fusedLocationClient.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, null)
            .addOnSuccessListener { location ->
                val link = if (location != null) "http://maps.google.com/?q=${location.latitude},${location.longitude}"
                else "EMERGENCY: GPS signal weak."
                sendFinalSMS(link)
            }
    }

    private fun sendFinalSMS(message: String) {
        val number = getSavedGuardianNumber()
        if (number.isEmpty()) {
            speakFeedback("No guardian number saved.")
            return
        }
        try {
            val smsManager = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) getSystemService(SmsManager::class.java)
            else SmsManager.getDefault()
            smsManager.sendTextMessage(number, null, "SOS Alert! Location: $message", null, null)
            speakFeedback("Emergency location shared")
        } catch (e: Exception) { Log.e("BlindNav", "SMS Error: ${e.message}") }
    }

    // --- SHARED PREFS ---
    fun saveGuardianNumber(number: String) = getSharedPreferences("BlindNavPrefs", MODE_PRIVATE).edit().putString("guardian_no", number).apply()
    fun getSavedGuardianNumber(): String = getSharedPreferences("BlindNavPrefs", MODE_PRIVATE).getString("guardian_no", "") ?: ""

    // --- VOICE & DETECTION LOGIC ---
    private fun setupSpeechRecognizer() {

        if (!SpeechRecognizer.isRecognitionAvailable(this)) return

        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)

        speechIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {

            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)

            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.US.toString())

        }

        speechRecognizer?.setRecognitionListener(object : RecognitionListener {

            override fun onReadyForSpeech(params: Bundle?) { isListening = true }

            override fun onEndOfSpeech() { isListening = false }

            override fun onError(error: Int) { isListening = false; speechRecognizer?.startListening(speechIntent) }

            override fun onResults(results: Bundle?) {

                results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.get(0)?.let { handleVoiceCommand(it.lowercase()) }

                speechRecognizer?.startListening(speechIntent)

            }

            override fun onBeginningOfSpeech() {}

            override fun onRmsChanged(rmsdB: Float) {}

            override fun onBufferReceived(buffer: ByteArray?) {}

            override fun onPartialResults(partialResults: Bundle?) {}

            override fun onEvent(eventType: Int, params: Bundle?) {}

        })

    }

    private fun handleVoiceCommand(command: String) {
        when {
            command.contains("check money") || command.contains("currency") -> {
                currentMode = AppMode.CURRENCY_DETECTION
                speakFeedback("Currency mode on.")
            }
            command.contains("detect objects") || command.contains("object mode") -> {
                currentMode = AppMode.OBJECT_DETECTION
                speakFeedback("Object mode on.")
            }
            command.contains("start") -> {
                isDetectionActive = true
                speakFeedback("Starting Detection")
            }
            command.contains("stop") -> {
                isDetectionActive = false
                speakFeedback("Stopping Detection")
                detectionsState = emptyList()
            }

            // --- Languages Section ---
            command.contains("telugu") -> {
                changeLanguage("Telugu", TranslateLanguage.TELUGU)
            }
            command.contains("english") -> {
                changeLanguage("English", TranslateLanguage.ENGLISH)
            }
            command.contains("hindi") -> {
                changeLanguage("Hindi", TranslateLanguage.HINDI)
            }
            command.contains("tamil") -> {
                changeLanguage("Tamil", TranslateLanguage.TAMIL)
            }
            command.contains("kannada") -> {
                changeLanguage("Kannada", TranslateLanguage.KANNADA)
            }
        }
    }

    // Code clean ga undadaniki ee helper function vaadandi
    private fun changeLanguage(name: String, code: String) {
        selectedLangName = name
        selectedLangCode = code
        speakFeedback("Language changed to $name")
        // Offline models download ayyeలా chustundi
        ensureModelDownloaded(code)
    }

    private fun speakFeedback(text: String) { tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null) }

    private fun triggerVibration(duration: Long) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) vibrator.vibrate(VibrationEffect.createOneShot(duration, VibrationEffect.DEFAULT_AMPLITUDE))
        else vibrator.vibrate(duration)
    }

    private fun ensureModelDownloaded(langCode: String) {
        if (langCode == TranslateLanguage.ENGLISH) return
        if (currentTranslatorLang != langCode) {
            translator?.close()
            translator = Translation.getClient(TranslatorOptions.Builder().setSourceLanguage(TranslateLanguage.ENGLISH).setTargetLanguage(langCode).build())
            currentTranslatorLang = langCode
        }
        val model = TranslateRemoteModel.Builder(langCode).build()
        RemoteModelManager.getInstance().isModelDownloaded(model).addOnSuccessListener { isDownloaded ->
            if (!isDownloaded) {
                isDownloadingModel = true
                speakFeedback("Downloading $selectedLangName model")
                translator?.downloadModelIfNeeded(DownloadConditions.Builder().build())?.addOnSuccessListener { isDownloadingModel = false; speakFeedback("$selectedLangName ready") }
            } else isDownloadingModel = false
        }
    }

    private fun translateAndSpeak(text: String) {
        if (text.contains("Rupee")) { tts?.speak("Detected ${text.replace("Rupee", "Rupees")}", TextToSpeech.QUEUE_FLUSH, null, null); return }
        if (selectedLangCode == TranslateLanguage.ENGLISH || isDownloadingModel) {
            tts?.language = Locale.US; tts?.speak("There is a $text", TextToSpeech.QUEUE_FLUSH, null, null); return
        }
        translator?.translate("There is a $text ahead")?.addOnSuccessListener { translated ->
            tts?.language = Locale(selectedLangCode); tts?.speak(translated, TextToSpeech.QUEUE_FLUSH, null, null)
        }
    }

    private fun detectObjects(imageProxy: ImageProxy) {
        if (!isDetectionActive) { imageProxy.close(); return }
        try {
            val bitmap = imageProxy.toBitmap() ?: return
            
            if (currentMode == AppMode.CURRENCY_DETECTION && currencyInterpreter != null) {
                // --- CUSTOM CURRENCY MODEL INFERENCE (Classification) ---
                val imageProcessor = ImageProcessor.Builder()
                    .add(Rot90Op(-imageProxy.imageInfo.rotationDegrees / 90))
                    .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                    .add(NormalizeOp(0f, 255f)).build()

                val tensorImage = TensorImage(DataType.FLOAT32).also { it.load(bitmap) }
                val outputBuffer = Array(1) { FloatArray(currencyLabels.size) }
                currencyInterpreter?.run(imageProcessor.process(tensorImage).buffer, outputBuffer)

                val scores = outputBuffer[0]
                val maxScoreIndex = scores.indices.maxByOrNull { scores[it] } ?: -1
                val maxScore = if (maxScoreIndex != -1) scores[maxScoreIndex] else 0f

                if (maxScore > 0.75f) {
                    val label = currencyLabels.getOrElse(maxScoreIndex) { "Currency" }
                    val result = DetectionResult(label, maxScore, RectF(0.1f, 0.1f, 0.9f, 0.9f))
                    
                    runOnUiThread {
                        detectionsState = listOf(result)
                        if (System.currentTimeMillis() - lastSpokenTime > speechInterval) {
                            translateAndSpeak(label)
                            lastSpokenTime = System.currentTimeMillis()
                        }
                    }
                }
            } else {
                // --- STANDALONE OBJECT DETECTION (YOLO) ---
                val imageProcessor = ImageProcessor.Builder()
                    .add(Rot90Op(-imageProxy.imageInfo.rotationDegrees / 90))
                    .add(ResizeOp(640, 640, ResizeOp.ResizeMethod.BILINEAR))
                    .add(NormalizeOp(0f, 255f)).build()

                val tensorImage = TensorImage(DataType.FLOAT32).also { it.load(bitmap) }
                val output = Array(1) { Array(84) { FloatArray(8400) } }
                interpreter?.run(imageProcessor.process(tensorImage).buffer, output)

                val results = mutableListOf<DetectionResult>()
                for (i in 0 until 8400) {
                    var maxScore = 0f; var classId = -1
                    for (c in 0 until 80) {
                        val score = output[0][c + 4][i]
                        if (score > maxScore) { maxScore = score; classId = c }
                    }
                    if (maxScore > 0.45f) {
                        val cx = output[0][0][i]; val cy = output[0][1][i]; val w = output[0][2][i]; val h = output[0][3][i]
                        results.add(DetectionResult(labels.getOrElse(classId) { "Object" }, maxScore, RectF((cx - w / 2f) / 640f, (cy - h / 2f) / 640f, (cx + w / 2f) / 640f, (cy + h / 2f) / 640f)))
                    }
                }
                val finalResults = applyNMS(results)
                if (finalResults.any { it.box.height() > 0.55f }) triggerVibration(200)

                runOnUiThread {
                    detectionsState = when (currentMode) {
                        AppMode.CURRENCY_DETECTION -> finalResults.filter { it.label.contains("Rupee") }
                        AppMode.OBJECT_DETECTION -> finalResults.filter { !it.label.contains("Rupee") }
                    }
                    if (detectionsState.isNotEmpty() && (System.currentTimeMillis() - lastSpokenTime > speechInterval)) {
                        translateAndSpeak(detectionsState.maxBy { it.score }.label)
                        lastSpokenTime = System.currentTimeMillis()
                    }
                }
            }
        } catch (e: Exception) { Log.e("BlindNav", "Error: ${e.message}") } finally { imageProxy.close() }
    }

    // --- SETUP HELPERS ---
    private fun setupInterpreter() {
        try {
            // Load Main YOLO Model
            val model = FileUtil.loadMappedFile(this, "yolo11n_int8.tflite")
            interpreter = Interpreter(model, Interpreter.Options().setNumThreads(4))
            labels = loadLabels()

            // Load Custom Currency Model (Classification)
            try {
                val currencyModel = FileUtil.loadMappedFile(this, "model_unquant.tflite")
                currencyInterpreter = Interpreter(currencyModel, Interpreter.Options().setNumThreads(2))
                // Reading labels from labels.txt instead of hardcoding to ensure consistency
                currencyLabels = FileUtil.loadLabels(this, "labels.txt").map { it.split(" ", limit = 2).last() }
            } catch (e: Exception) {
                Log.e("BlindNav", "Currency Model Load Fail: ${e.message}")
            }
        } catch (e: Exception) { Log.e("BlindNav", "Model Load Fail") }
    }

    private fun setupTTS() { tts = TextToSpeech(this) { if (it == TextToSpeech.SUCCESS) { tts?.language = Locale.US; tts?.setSpeechRate(0.8f) } } }
    private fun setupLightSensor() { sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager; lightSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LIGHT) }
    override fun onResume() { super.onResume(); lightSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_UI) } }
    override fun onPause() { super.onPause(); sensorManager.unregisterListener(this) }
    override fun onSensorChanged(event: SensorEvent?) { if (event?.sensor?.type == Sensor.TYPE_LIGHT) isEnvironmentDark = event.values[0] < 15f }
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun loadLabels(): List<String> {
        return try {
            // Primarily try to use the YOLO labels from the shared resource if possible, 
            // but providing the standard COCO list as fallback for the main YOLO model.
            listOf("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "10 Rupee", "20 Rupee", "50 Rupee", "100 Rupee", "200 Rupee", "500 Rupee")
        } catch (e: Exception) {
            emptyList()
        }
    }

    private fun applyNMS(results: List<DetectionResult>): List<DetectionResult> {
        val sorted = results.sortedByDescending { it.score }; val selected = mutableListOf<DetectionResult>()
        for (res in sorted) {
            if (selected.none { calculateIoU(res.box, it.box) > 0.45f }) { selected.add(res); if (selected.size >= 5) break }
        }
        return selected
    }

    private fun calculateIoU(b1: RectF, b2: RectF): Float {
        val intArea = maxOf(0f, minOf(b1.right, b2.right) - maxOf(b1.left, b2.left)) * maxOf(0f, minOf(b1.bottom, b2.bottom) - maxOf(b1.top, b2.top))
        val uArea = (b1.width() * b1.height()) + (b2.width() * b2.height()) - intArea
        return if (uArea > 0) intArea / uArea else 0f
    }

    override fun onDestroy() { 
        super.onDestroy()
        tts?.shutdown()
        interpreter?.close()
        currencyInterpreter?.close()
        speechRecognizer?.destroy()
        translator?.close() 
    }
}

// --- UI COMPONENTS ---
@Composable
fun ObjectDetectionScreen(detections: List<DetectionResult>, isDark: Boolean, onAnalyze: (ImageProxy) -> Unit) {
    val lifecycleOwner = LocalLifecycleOwner.current
    val context = LocalContext.current as MainActivity
    var showSettings by remember { mutableStateOf(false) }
    var camera by remember { mutableStateOf<Camera?>(null) }

    LaunchedEffect(isDark) { camera?.cameraControl?.enableTorch(isDark) }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(modifier = Modifier.fillMaxSize(), factory = { ctx ->
            PreviewView(ctx).apply {
                ProcessCameraProvider.getInstance(ctx).addListener({
                    val provider = ProcessCameraProvider.getInstance(ctx).get()
                    val preview = Preview.Builder().build().also { it.setSurfaceProvider(surfaceProvider) }
                    val analysis = ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888).build().also { it.setAnalyzer(Executors.newSingleThreadExecutor()) { proxy -> onAnalyze(proxy) } }
                    provider.unbindAll(); camera = provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
                }, ContextCompat.getMainExecutor(ctx))
            }
        })
        Canvas(modifier = Modifier.fillMaxSize()) {
            drawIntoCanvas { canvas ->
                detections.forEach {
                    val rect = RectF(it.box.left * size.width, it.box.top * size.height, it.box.right * size.width, it.box.bottom * size.height)
                    canvas.nativeCanvas.drawRect(rect, Paint().apply { color = android.graphics.Color.GREEN; style = Paint.Style.STROKE; strokeWidth = 8f })
                    canvas.nativeCanvas.drawText(it.label, rect.left, rect.top - 10f, Paint().apply { color = android.graphics.Color.WHITE; textSize = 40f })
                }
            }
        }
        IconButton(onClick = { showSettings = true }, modifier = Modifier.align(Alignment.TopEnd).padding(16.dp)) {
            Icon(Icons.Default.Settings, contentDescription = "Settings", tint = Color.White, modifier = Modifier.size(36.dp))
        }
        if (showSettings) {
            MergedSettingsDialog(
                currentNumber = context.getSavedGuardianNumber(),
                currentLang = context.getSavedGuardianNumber(), // Logic used to trigger dialog
                onDismiss = { showSettings = false },
                onSaveNumber = { context.saveGuardianNumber(it) },
                onLangChange = { /* Handled via voice or manual logic if needed */ }
            )
        }
    }
}

@Composable
fun MergedSettingsDialog(currentNumber: String, currentLang: String, onDismiss: () -> Unit, onSaveNumber: (String) -> Unit, onLangChange: (String) -> Unit) {
    var numberText by remember { mutableStateOf(currentNumber) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Settings") },
        text = {
            Column {
                Text("Guardian Mobile Number:")
                TextField(value = numberText, onValueChange = { numberText = it }, modifier = Modifier.fillMaxWidth())
                Spacer(modifier = Modifier.height(16.dp))
                Text("Note: Use voice commands like 'Change language to Telugu' to switch output language.")
            }
        },
        confirmButton = { Button(onClick = { onSaveNumber(numberText); onDismiss() }) { Text("Save All") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } }
    )
}

@Composable
fun VoiceStatusOverlay(isListening: Boolean, isActive: Boolean, mode: AppMode) {
    Box(modifier = Modifier.fillMaxSize().padding(bottom = 24.dp), contentAlignment = Alignment.BottomCenter) {
        Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.background(Color.Black.copy(0.6f), MaterialTheme.shapes.medium).padding(12.dp)) {
            if (isListening) Text("Listening...", color = Color.Cyan)
            Text("MODE: ${mode.name}", color = Color.White)
            Text(if (isActive) "SYSTEM ON" else "SYSTEM OFF", color = if (isActive) Color.Green else Color.Red)
        }
    }
}

private fun checkAndEnableGPS(context: Context) {
    val client = LocationServices.getSettingsClient(context)
    val builder = com.google.android.gms.location.LocationSettingsRequest.Builder().addLocationRequest(com.google.android.gms.location.LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 10000).build())
    client.checkLocationSettings(builder.build()).addOnFailureListener { e ->
        if (e is com.google.android.gms.common.api.ResolvableApiException) {
            try { e.startResolutionForResult(context as android.app.Activity, 1001) } catch (err: Exception) { Log.e("BlindNav", "GPS prompt error") }
        }
    }
}
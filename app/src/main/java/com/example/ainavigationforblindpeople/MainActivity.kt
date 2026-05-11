package com.example.ainavigationforblindpeople

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.view.KeyEvent
import android.graphics.Paint
import android.graphics.RectF
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.AudioManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.telephony.SmsManager
import android.util.Log
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
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.lifecycleScope
import com.example.ainavigationforblindpeople.ui.theme.AINavigationForBlindPeopleTheme
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.common.model.RemoteModelManager
import com.google.mlkit.nl.translate.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
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

    // State for UI and Haptics
    private var _alertLevel by mutableStateOf(0)
    var alertLevel: Int
        get() = _alertLevel
        set(value) {
            _alertLevel = value
            Log.w("ALERT", "Alert level set to: $value")
        }

    // SOS Variables
    private var pressStartTime: Long = 0
    private val sosThreshold = 3000L
    private var sosTriggered = false

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
    private var isDetectionActive by mutableStateOf(true)
    private var currentMode by mutableStateOf(AppMode.OBJECT_DETECTION)

    // Voice & Vibration
    private var speechRecognizer: SpeechRecognizer? = null
    private lateinit var speechIntent: Intent
    private var isListening by mutableStateOf(false)

    private var vibrator: android.os.Vibrator? = null
    private var camera: Camera? = null
    private var torchEnabled = false
    private var lastVibrationTime = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        vibrator = getSystemService(android.os.Vibrator::class.java)

        Handler(Looper.getMainLooper()).postDelayed({
            Log.w("TEST", "Testing vibration...")
            triggerVibration(500L)
            speakFeedback("Vibration test")
        }, 2000)

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
                            ObjectDetectionScreen(detectionsState, alertLevel, isEnvironmentDark) { imageProxy ->
                                processImageProxy(imageProxy)
                            }
                            VoiceStatusOverlay(isListening, isDetectionActive, currentMode)
                        }
                    }
                }
            }
        }
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean {
        when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                Log.d("SOS", "Volume Down pressed")
                if (event.repeatCount == 0) {
                    pressStartTime = System.currentTimeMillis()
                    sosTriggered = false
                    Handler(Looper.getMainLooper()).postDelayed({
                        if (!sosTriggered && System.currentTimeMillis() - pressStartTime >= sosThreshold) {
                            triggerSOS()
                            sosTriggered = true
                        }
                    }, sosThreshold)
                }
                return true
            }
            KeyEvent.KEYCODE_VOLUME_UP -> {
                return super.onKeyDown(keyCode, event)
            }
        }
        return super.onKeyDown(keyCode, event)
    }

    override fun onKeyUp(keyCode: Int, event: KeyEvent): Boolean {
        when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                if (!sosTriggered) {
                    val audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
                    audioManager.adjustVolume(AudioManager.ADJUST_LOWER, AudioManager.FLAG_SHOW_UI)
                }
                pressStartTime = 0L
                return true
            }
        }
        return super.onKeyUp(keyCode, event)
    }

    @SuppressLint("MissingPermission")
    private fun triggerSOS() {
        Log.w("SOS", "!!! SOS TRIGGERED !!!")
        triggerVibration(1000L)

        val number = getSavedGuardianNumber()
        if (number.isEmpty()) {
            speakFeedback("No guardian number saved")
            Toast.makeText(this, "No guardian number saved", Toast.LENGTH_LONG).show()
            return
        }

        speakFeedback("Emergency! Sending SOS")

        val fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        fusedLocationClient.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, null)
            .addOnSuccessListener { location ->
                val message = if (location != null) {
                    "EMERGENCY SOS! Location: https://maps.google.com/?q=${location.latitude},${location.longitude}"
                } else {
                    "EMERGENCY SOS! Need immediate help."
                }
                sendSMS(number, message)
            }
            .addOnFailureListener {
                sendSMS(number, "EMERGENCY SOS! Need immediate help.")
            }
    }

    private fun sendSMS(phoneNumber: String, message: String) {
        try {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.SEND_SMS) != PackageManager.PERMISSION_GRANTED) {
                speakFeedback("SMS permission not granted")
                return
            }

            val smsManager = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                getSystemService(SmsManager::class.java)
            } else {
                @Suppress("DEPRECATION")
                SmsManager.getDefault()
            }

            smsManager.sendTextMessage(phoneNumber, null, message, null, null)
            Log.w("SOS", "SMS sent successfully")
            speakFeedback("SOS message sent")
            Toast.makeText(this, "SOS sent", Toast.LENGTH_LONG).show()
        } catch (e: Exception) {
            Log.e("SOS", "Failed to send SMS: ${e.message}")
            speakFeedback("Failed to send SOS")
        }
    }

    fun saveGuardianNumber(number: String) {
        getSharedPreferences("BlindNavPrefs", MODE_PRIVATE).edit().putString("guardian_no", number).apply()
    }

    fun getSavedGuardianNumber(): String {
        return getSharedPreferences("BlindNavPrefs", MODE_PRIVATE).getString("guardian_no", "") ?: ""
    }

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
            override fun onError(error: Int) {
                isListening = false
                Handler(Looper.getMainLooper()).postDelayed({
                    speechRecognizer?.startListening(speechIntent)
                }, 1000)
            }
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
            command.contains("currency") || command.contains("check money") -> {
                currentMode = AppMode.CURRENCY_DETECTION
                speakFeedback("Currency mode on")
            }
            command.contains("object") || command.contains("detect objects") -> {
                currentMode = AppMode.OBJECT_DETECTION
                speakFeedback("Object mode on")
            }
            command.contains("start") -> {
                isDetectionActive = true
                speakFeedback("Detection started")
                if (isEnvironmentDark) enableTorch(true)
            }
            command.contains("stop") -> {
                isDetectionActive = false
                speakFeedback("Detection stopped")
                detectionsState = emptyList()
                enableTorch(false)
            }
            command.contains("telugu") -> changeLanguage("Telugu", TranslateLanguage.TELUGU)
            command.contains("english") -> changeLanguage("English", TranslateLanguage.ENGLISH)
            command.contains("hindi") -> changeLanguage("Hindi", TranslateLanguage.HINDI)
            command.contains("tamil") -> changeLanguage("Tamil", TranslateLanguage.TAMIL)
            command.contains("kannada") -> changeLanguage("Kannada", TranslateLanguage.KANNADA)
        }
    }

    private fun changeLanguage(name: String, code: String) {
        selectedLangName = name
        selectedLangCode = code
        speakFeedback("Language changed to $name")
        ensureModelDownloaded(code)
    }

    fun speakFeedback(text: String) {
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
        Log.d("TTS", text)
    }

    private fun triggerVibration(duration: Long) {
        val v = vibrator ?: return
        if (!v.hasVibrator()) return

        val currentTime = System.currentTimeMillis()
        if (currentTime - lastVibrationTime < 200) return
        lastVibrationTime = currentTime

        Handler(Looper.getMainLooper()).post {
            try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    v.vibrate(android.os.VibrationEffect.createOneShot(duration, android.os.VibrationEffect.DEFAULT_AMPLITUDE))
                } else {
                    @Suppress("DEPRECATION")
                    v.vibrate(duration)
                }
            } catch (e: Exception) { }
        }
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
                translator?.downloadModelIfNeeded(DownloadConditions.Builder().build())?.addOnSuccessListener {
                    isDownloadingModel = false
                    speakFeedback("$selectedLangName ready")
                }
            } else isDownloadingModel = false
        }
    }

    private fun translateAndSpeak(text: String) {
        val lowerText = text.lowercase()
        if (lowerText.contains("rupee")) {
            var speechText = text
            when {
                lowerText.contains("ten") -> speechText = "10 Rupees"
                lowerText.contains("twenty") -> speechText = "20 Rupees"
                lowerText.contains("fifty") -> speechText = "50 Rupees"
                lowerText.contains("one hundred") -> speechText = "100 Rupees"
                lowerText.contains("two hundred") -> speechText = "200 Rupees"
                lowerText.contains("five hundred") -> speechText = "500 Rupees"
            }
            tts?.speak("Detected $speechText", TextToSpeech.QUEUE_FLUSH, null, null)
            return
        }
        if (selectedLangCode == TranslateLanguage.ENGLISH || isDownloadingModel) {
            tts?.language = Locale.US
            tts?.speak("There is a $text", TextToSpeech.QUEUE_FLUSH, null, null)
            return
        }
        translator?.translate("There is a $text ahead")?.addOnSuccessListener { translated ->
            tts?.language = Locale(selectedLangCode)
            tts?.speak(translated, TextToSpeech.QUEUE_FLUSH, null, null)
        }
    }

    private fun processImageProxy(imageProxy: ImageProxy) {
        if (!isDetectionActive) { imageProxy.close(); return }

        if (currentMode == AppMode.CURRENCY_DETECTION && currencyInterpreter != null) {
            processCurrencyProxy(imageProxy)
            return
        }

        val currentInterpreter = interpreter ?: run { imageProxy.close(); return }

        imageProxy.use { proxy ->
            try {
                val bitmap = proxy.toBitmap()
                val rotationDegrees = proxy.imageInfo.rotationDegrees

                val imageProcessor = ImageProcessor.Builder()
                    .add(Rot90Op(-rotationDegrees / 90))
                    .add(ResizeOp(640, 640, ResizeOp.ResizeMethod.BILINEAR))
                    .add(NormalizeOp(0f, 255f))
                    .build()

                val tensorImage = TensorImage(DataType.FLOAT32).also { it.load(bitmap) }
                val processedImage = imageProcessor.process(tensorImage)

                val output = Array(1) { Array(84) { FloatArray(8400) } }
                currentInterpreter.run(processedImage.buffer, output)

                val results = mutableListOf<DetectionResult>()

                for (i in 0 until 8400) {
                    var maxScore = 0f
                    var classId = -1
                    for (c in 0 until 80) {
                        val score = output[0][c + 4][i]
                        if (score > maxScore) {
                            maxScore = score
                            classId = c
                        }
                    }

                    if (maxScore > 0.4f && classId >= 0) {
                        val cx = output[0][0][i]
                        val cy = output[0][1][i]
                        val w = output[0][2][i]
                        val h = output[0][3][i]

                        val left = (cx - w / 2f).coerceIn(0f, 1f)
                        val top = (cy - h / 2f).coerceIn(0f, 1f)
                        val right = (cx + w / 2f).coerceIn(0f, 1f)
                        val bottom = (cy + h / 2f).coerceIn(0f, 1f)

                        val label = labels.getOrElse(classId) { "object" }
                        val box = RectF(left, top, right, bottom)

                        results.add(DetectionResult(label, maxScore, box))
                    }
                }

                val finalResults = applyNMS(results)

                lifecycleScope.launch(Dispatchers.Main) {
                    val filteredResults = when (currentMode) {
                        AppMode.CURRENCY_DETECTION -> finalResults.filter { it.label.contains("Rupee") }
                        AppMode.OBJECT_DETECTION -> finalResults
                    }

                    if (filteredResults.isNotEmpty()) {
                        updateProximityAndVibration(filteredResults)
                        detectionsState = filteredResults

                        if (System.currentTimeMillis() - lastSpokenTime > speechInterval) {
                            val best = filteredResults.maxByOrNull { it.score }
                            best?.let {
                                translateAndSpeak(it.label)
                                lastSpokenTime = System.currentTimeMillis()
                            }
                        }
                    } else {
                        detectionsState = emptyList()
                    }
                }
            } catch (e: Exception) { Log.e("DETECTOR", "Error: ${e.message}") }
        }
    }

    private fun processCurrencyProxy(imageProxy: ImageProxy) {
        val currInterpreter = currencyInterpreter ?: run { imageProxy.close(); return }
        imageProxy.use { proxy ->
            try {
                val bitmap = proxy.toBitmap()
                val rotationDegrees = proxy.imageInfo.rotationDegrees
                val imageProcessor = ImageProcessor.Builder()
                    .add(Rot90Op(-rotationDegrees / 90))
                    .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                    .add(NormalizeOp(127.5f, 127.5f))
                    .build()

                val tensorImage = TensorImage(DataType.FLOAT32).also { it.load(bitmap) }
                val processedImage = imageProcessor.process(tensorImage)

                val output = Array(1) { FloatArray(currencyLabels.size) }
                currInterpreter.run(processedImage.buffer, output)

                var maxIdx = -1
                var maxScore = 0f
                for (i in output[0].indices) {
                    if (output[0][i] > maxScore) {
                        maxScore = output[0][i]
                        maxIdx = i
                    }
                }

                if (maxIdx != -1 && maxScore > 0.8f) {
                    val label = currencyLabels[maxIdx]
                    if (label.lowercase() != "background") {
                        val result = DetectionResult(label, maxScore, RectF(0.1f, 0.1f, 0.9f, 0.9f))
                        lifecycleScope.launch(Dispatchers.Main) {
                            detectionsState = listOf(result)
                            if (System.currentTimeMillis() - lastSpokenTime > speechInterval) {
                                translateAndSpeak(label)
                                lastSpokenTime = System.currentTimeMillis()
                            }
                        }
                        return
                    }
                }
                lifecycleScope.launch(Dispatchers.Main) { detectionsState = emptyList() }
            } catch (e: Exception) { Log.e("DETECTOR", "Currency error: ${e.message}") }
        }
    }

    private fun updateProximityAndVibration(results: List<DetectionResult>) {
        if (results.isEmpty()) { alertLevel = 0; return }

        var largestSize = 0f
        var closestLabel = "object"
        var isVehicleNearby = false

        for (res in results) {
            val size = res.box.width() * res.box.height()
            val isVehicle = res.label in listOf("car", "bus", "truck", "motorcycle", "bicycle")
            if (isVehicle) isVehicleNearby = true
            if (size > largestSize) {
                largestSize = size
                closestLabel = res.label
            }
        }

        alertLevel = when {
            largestSize > 0.35f -> 2
            largestSize > 0.15f || isVehicleNearby -> 1
            else -> 0
        }

        when {
            largestSize > 0.35f -> {
                triggerVibration(600L)
                if (System.currentTimeMillis() - lastSpokenTime > 3000) {
                    speakFeedback("Stop! $closestLabel is very close")
                    lastSpokenTime = System.currentTimeMillis()
                }
            }
            largestSize > 0.15f -> {
                triggerVibration(300L)
                if (System.currentTimeMillis() - lastSpokenTime > 4000) {
                    speakFeedback("$closestLabel is ahead")
                    lastSpokenTime = System.currentTimeMillis()
                }
            }
            isVehicleNearby -> triggerVibration(200L)
        }
    }

    private fun setupInterpreter() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // Load Object Detection Model
                val model = FileUtil.loadMappedFile(this@MainActivity, "yolo11n_int8.tflite")
                interpreter = Interpreter(model, Interpreter.Options().setNumThreads(4))
                labels = loadLabels()

                // Load Currency Detection Model
                val currencyModel = FileUtil.loadMappedFile(this@MainActivity, "model_unquant.tflite")
                currencyInterpreter = Interpreter(currencyModel, Interpreter.Options().setNumThreads(4))
                currencyLabels = try {
                    assets.open("labels.txt").bufferedReader().useLines { lines ->
                        lines.map { it.substringAfter(" ").trim() }.toList()
                    }
                } catch (e: Exception) {
                    listOf("Background", "ten rupee notes", "twenty rupee notes", "fifty rupee notes", "one hundred rupees", "two hundred rupees", "five hundred rupees")
                }

                Log.d("DETECTOR", "Models loaded successfully")
                speakFeedback("Systems ready")
            } catch (e: Exception) {
                Log.e("DETECTOR", "Model loading error: ${e.message}")
                speakFeedback("Error loading models")
            }
        }
    }

    fun startCamera(previewView: PreviewView, lifecycleOwner: LifecycleOwner, onAnalyze: (ImageProxy) -> Unit) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val provider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(Executors.newSingleThreadExecutor()) { proxy -> onAnalyze(proxy) } }

            try {
                provider.unbindAll()
                camera = provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            } catch (e: Exception) { Log.e("DETECTOR", "Binding failed", e) }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun setupTTS() {
        tts = TextToSpeech(this) {
            if (it == TextToSpeech.SUCCESS) {
                tts?.language = Locale.US
                tts?.setSpeechRate(0.8f)
                speakFeedback("App ready")
            }
        }
    }

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
            val lux = event.values[0]
            isEnvironmentDark = lux < 15f
            if (isDetectionActive) {
                if (lux < 10f && !torchEnabled) enableTorch(true)
                else if (lux > 20f && torchEnabled) enableTorch(false)
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun loadLabels() = listOf("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "10 Rupee", "20 Rupee", "50 Rupee", "100 Rupee", "200 Rupee", "500 Rupee")

    private fun applyNMS(results: List<DetectionResult>): List<DetectionResult> {
        val sorted = results.sortedByDescending { it.score }
        val selected = mutableListOf<DetectionResult>()
        for (res in sorted) {
            if (selected.none { calculateIoU(res.box, it.box) > 0.45f }) {
                selected.add(res)
                if (selected.size >= 5) break
            }
        }
        return selected
    }

    private fun calculateIoU(b1: RectF, b2: RectF): Float {
        val intArea = maxOf(0f, minOf(b1.right, b2.right) - maxOf(b1.left, b2.left)) *
                maxOf(0f, minOf(b1.bottom, b2.bottom) - maxOf(b1.top, b2.top))
        val uArea = (b1.width() * b1.height()) + (b2.width() * b2.height()) - intArea
        return if (uArea > 0) intArea / uArea else 0f
    }

    override fun onDestroy() {
        super.onDestroy()
        tts?.shutdown()
        interpreter?.close()
        speechRecognizer?.destroy()
        translator?.close()
        enableTorch(false)
    }

    private fun enableTorch(enable: Boolean) {
        try {
            camera?.cameraControl?.enableTorch(enable)
            torchEnabled = enable
        } catch (e: Exception) { }
    }
}

// --- UI COMPONENTS ---
@Composable
fun ObjectDetectionScreen(detections: List<DetectionResult>, alertLevel: Int, isDark: Boolean, onAnalyze: (ImageProxy) -> Unit) {
    val lifecycleOwner = LocalLifecycleOwner.current
    val context = LocalContext.current as MainActivity
    var showSettings by remember { mutableStateOf(false) }
    var camera by remember { mutableStateOf<Camera?>(null) }

    LaunchedEffect(isDark) { camera?.cameraControl?.enableTorch(isDark) }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(modifier = Modifier.fillMaxSize(), factory = { ctx ->
            PreviewView(ctx).apply {
                context.startCamera(this, lifecycleOwner) { proxy ->
                    onAnalyze(proxy)
                }
            }
        })

        val overlayColor = when (alertLevel) {
            2 -> Color.Red.copy(alpha = 0.5f)
            1 -> Color.Yellow.copy(alpha = 0.4f)
            else -> Color.Transparent
        }
        Box(modifier = Modifier.fillMaxSize().background(overlayColor))

        Canvas(modifier = Modifier.fillMaxSize()) {
            drawIntoCanvas { canvas ->
                detections.forEach {
                    val rect = android.graphics.RectF(
                        it.box.left * size.width,
                        it.box.top * size.height,
                        it.box.right * size.width,
                        it.box.bottom * size.height
                    )
                    canvas.nativeCanvas.drawRect(rect, Paint().apply {
                        color = android.graphics.Color.GREEN
                        style = Paint.Style.STROKE
                        strokeWidth = 8f
                    })
                    canvas.nativeCanvas.drawText(
                        "${it.label} ${(it.score * 100).toInt()}%",
                        rect.left, rect.top - 10f,
                        Paint().apply { color = android.graphics.Color.WHITE; textSize = 40f }
                    )
                }
            }
        }

        IconButton(onClick = { showSettings = true }, modifier = Modifier.align(Alignment.TopEnd).padding(16.dp)) {
            Icon(Icons.Default.Settings, contentDescription = "Settings", tint = Color.White, modifier = Modifier.size(36.dp))
        }

        if (showSettings) {
            var numberText by remember { mutableStateOf(context.getSavedGuardianNumber()) }
            AlertDialog(
                onDismissRequest = { showSettings = false },
                title = { Text("Settings") },
                text = {
                    Column(modifier = Modifier.fillMaxWidth().padding(8.dp)) {
                        Text("Guardian Mobile Number:")
                        TextField(value = numberText, onValueChange = { numberText = it }, modifier = Modifier.fillMaxWidth(), placeholder = { Text("Enter 10-digit number") })
                        Spacer(modifier = Modifier.height(16.dp))
                        Text("Voice: 'Currency mode', 'Object mode', 'Start', 'Stop'")
                        Text("Languages: 'Telugu', 'English', 'Hindi', 'Tamil', 'Kannada'")
                        Text("SOS: Hold Volume Down 3 seconds", color = Color.Red)
                    }
                },
                confirmButton = {
                    Button(onClick = {
                        if (numberText.length >= 10) context.saveGuardianNumber(numberText)
                        showSettings = false
                    }) { Text("Save") }
                },
                dismissButton = { TextButton(onClick = { showSettings = false }) { Text("Cancel") } }
            )
        }
    }
}

@Composable
fun VoiceStatusOverlay(isListening: Boolean, isActive: Boolean, mode: AppMode) {
    Box(modifier = Modifier.fillMaxSize().padding(bottom = 24.dp), contentAlignment = Alignment.BottomCenter) {
        Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.background(Color.Black.copy(alpha = 0.6f)).padding(12.dp)) {
            if (isListening) Text("🎤 Listening...", color = Color.Cyan)
            Text("Mode: ${mode.name}", color = Color.White)
            Text(if (isActive) "🟢 ACTIVE" else "🔴 PAUSED", color = if (isActive) Color.Green else Color.Red)
        }
    }
}

private fun checkAndEnableGPS(context: Context) {
    val client = LocationServices.getSettingsClient(context)
    val builder = com.google.android.gms.location.LocationSettingsRequest.Builder()
        .addLocationRequest(com.google.android.gms.location.LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 10000).build())
    client.checkLocationSettings(builder.build()).addOnFailureListener { e ->
        if (e is com.google.android.gms.common.api.ResolvableApiException) {
            try { e.startResolutionForResult(context as android.app.Activity, 1001) } catch (err: Exception) { }
        }
    }
}
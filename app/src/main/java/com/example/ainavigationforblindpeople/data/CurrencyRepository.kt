package com.example.ainavigationforblindpeople.data

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.DataType
import java.io.BufferedReader
import java.io.InputStreamReader

class CurrencyRepository(private val context: Context) {
    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()
    private var isModelLoaded = false

    // Confidence threshold for currency detection (higher for accuracy)
    private val confidenceThreshold = 0.70f

    suspend fun loadModel() {
        return kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
            try {
                // Load the model
                val model = FileUtil.loadMappedFile(context, "model_unquant.tflite")
                interpreter = Interpreter(model, Interpreter.Options().setNumThreads(2))

                // Load labels from labels.txt
                loadLabels()

                isModelLoaded = true
                Log.d("CurrencyRepo", "✅ Currency model loaded successfully with ${labels.size} labels")
            } catch (e: Exception) {
                Log.e("CurrencyRepo", "❌ Failed to load currency model: ${e.message}")
                isModelLoaded = false
            }
        }
    }

    private fun loadLabels() {
        try {
            val inputStream = context.assets.open("labels.txt")
            val reader = BufferedReader(InputStreamReader(inputStream))
            labels = reader.readLines()
            reader.close()
            Log.d("CurrencyRepo", "Loaded ${labels.size} currency labels: $labels")
        } catch (e: Exception) {
            Log.e("CurrencyRepo", "Failed to load labels: ${e.message}")
            // Fallback labels
            labels = listOf("10 Rupee", "20 Rupee", "50 Rupee", "100 Rupee", "200 Rupee", "500 Rupee")
        }
    }

    fun detectCurrency(
        bitmap: Bitmap,
        rotationDegrees: Int
    ): DetectionResult? {
        if (!isModelLoaded) {
            Log.d("CurrencyRepo", "Model not loaded yet")
            return null
        }

        val currentInterpreter = interpreter ?: return null

        try {
            // Prepare image processor (model expects 224x224 input)
            val imageProcessor = ImageProcessor.Builder()
                .add(Rot90Op(-rotationDegrees / 90))
                .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(127.5f, 127.5f))
                .build()

            // Convert to TensorImage
            val tensorImage = TensorImage(DataType.FLOAT32).also { it.load(bitmap) }
            val processedImage = imageProcessor.process(tensorImage)

            // Create output buffer
            val outputBuffer = Array(1) { FloatArray(labels.size) }

            // Run inference
            currentInterpreter.run(processedImage.buffer, outputBuffer)

            // Process results
            val scores = outputBuffer[0]
            var maxScore = 0f
            var maxIndex = -1

            for (i in scores.indices) {
                if (scores[i] > maxScore) {
                    maxScore = scores[i]
                    maxIndex = i
                }
            }

            Log.d("CurrencyRepo", "Best score: ${String.format("%.3f", maxScore)} at index $maxIndex")

            if (maxScore >= confidenceThreshold && maxIndex >= 0 && maxIndex < labels.size) {
                val label = labels[maxIndex]
                Log.w("CurrencyRepo", "🎯 Currency detected: $label (${String.format("%.0f", maxScore * 100)}%)")
                // Return detection with center box
                return DetectionResult(label, maxScore, RectF(0.2f, 0.2f, 0.8f, 0.8f))
            }

        } catch (e: Exception) {
            Log.e("CurrencyRepo", "Detection failed: ${e.message}")
        }

        return null
    }

    fun isReady(): Boolean = isModelLoaded && interpreter != null

    fun close() {
        interpreter?.close()
        interpreter = null
        isModelLoaded = false
    }
}

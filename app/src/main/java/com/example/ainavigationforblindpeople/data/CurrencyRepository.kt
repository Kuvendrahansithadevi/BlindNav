package com.example.ainavigationforblindpeople.data

import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op

class CurrencyRepository(
    private val context: android.content.Context,
    private val currencyLabels: List<String>
) {
    private var currencyInterpreter: Interpreter? = null
    private val confidenceThreshold = 0.80f

    suspend fun loadModel() {
        return kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
            try {
                val model = FileUtil.loadMappedFile(context, "currency_model.tflite")
                currencyInterpreter = Interpreter(model, Interpreter.Options().setNumThreads(2))
                Log.d("CurrencyRepo", "✅ Currency model loaded successfully")
            } catch (e: Exception) {
                Log.e("CurrencyRepo", "❌ Failed to load currency model: ${e.message}")
                currencyInterpreter = null
            }
        }
    }

    fun detectCurrency(
        bitmap: Bitmap,
        rotationDegrees: Int
    ): DetectionResult? {
        val interpreter = currencyInterpreter
        if (interpreter == null) {
            Log.d("CurrencyRepo", "Currency interpreter not available")
            return null
        }

        try {
            // Prepare image processor
            val imageProcessor = ImageProcessor.Builder()
                .add(Rot90Op(-rotationDegrees / 90))
                .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f))
                .build()

            // Convert to TensorImage
            val tensorImage = TensorImage(DataType.FLOAT32).also { it.load(bitmap) }
            val processedImage = imageProcessor.process(tensorImage)

            // Create output buffer
            val outputBuffer = Array(1) { FloatArray(currencyLabels.size) }

            // Run inference
            interpreter.run(processedImage.buffer, outputBuffer)

            // Process results
            val scores = outputBuffer[0]
            val maxScoreIndex = scores.indices.maxByOrNull { scores[it] } ?: -1
            val maxScore = if (maxScoreIndex != -1) scores[maxScoreIndex] else 0f

            Log.d("CurrencyRepo", "Detection score: ${String.format("%.3f", maxScore)}, Index: $maxScoreIndex")

            if (maxScore >= confidenceThreshold && maxScoreIndex >= 0 && maxScoreIndex < currencyLabels.size) {
                val label = currencyLabels[maxScoreIndex]
                // Return detection with full screen bounding box
                return DetectionResult(label, maxScore, RectF(0.1f, 0.1f, 0.9f, 0.9f))
            }

        } catch (e: Exception) {
            Log.e("CurrencyRepo", "Currency detection failed: ${e.message}")
        }

        return null
    }

    fun close() {
        currencyInterpreter?.close()
        currencyInterpreter = null
    }

    fun isAvailable(): Boolean = currencyInterpreter != null
}
package com.example.ainavigationforblindpeople

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.*
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.example.ainavigationforblindpeople.data.DetectionResult
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class ObstacleAnalyzer(
    private val context: Context,
    private val interpreter: Interpreter,
    private val labels: List<String>,
    private val onDetectionUpdate: (List<DetectionResult>, Boolean) -> Unit
) : ImageAnalysis.Analyzer {

    private val vibrator = context.getSystemService(Vibrator::class.java)
    private val handler = Handler(Looper.getMainLooper())
    
    // Thresholds for "2-meter" alert based on object type (normalized box height)
    // Larger objects (person) have larger thresholds than small ones (bottle)
    private val distanceThresholds = mapOf(
        "person" to 0.45f,
        "car" to 0.40f,
        "chair" to 0.35f,
        "bottle" to 0.20f
    )
    private val defaultThreshold = 0.30f

    override fun analyze(imageProxy: ImageProxy) {
        try {
            val bitmap = imageProxy.toBitmap() ?: return
            
            // Image Pre-processing
            val imageProcessor = ImageProcessor.Builder()
                .add(Rot90Op(-imageProxy.imageInfo.rotationDegrees / 90))
                .add(ResizeOp(640, 640, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f))
                .build()

            val tensorImage = TensorImage(DataType.FLOAT32).also { it.load(bitmap) }
            val processedImage = imageProcessor.process(tensorImage)

            // YOLOv8/11 Output Format: [1, 84, 8400]
            val output = Array(1) { Array(84) { FloatArray(8400) } }
            interpreter.run(processedImage.buffer, output)

            val detections = mutableListOf<DetectionResult>()
            var isDangerDetected = false

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

                if (maxScore > 0.45f) {
                    val label = labels.getOrElse(classId) { "Object" }
                    val cx = output[0][0][i]
                    val cy = output[0][1][i]
                    val w = output[0][2][i]
                    val h = output[0][3][i]

                    val box = RectF(
                        (cx - w / 2f) / 640f,
                        (cy - h / 2f) / 640f,
                        (cx + w / 2f) / 640f,
                        (cy + h / 2f) / 640f
                    )

                    detections.add(DetectionResult(label, maxScore, box))

                    // Distance logic: Check if object is within "2-meter" threshold
                    val threshold = distanceThresholds[label] ?: defaultThreshold
                    if (box.height() > threshold) {
                        isDangerDetected = true
                    }
                }
            }
            val startTime = System.currentTimeMillis()

             // ...  Inference logic ikkada  ...

             val inferenceTime = System.currentTimeMillis() - startTime
             Log.d("Performance_Benchmarking", "Inference Latency: ${inferenceTime}ms")

            if (inferenceTime < 100) {
            Log.i("Performance_Benchmarking", "Latency Requirement Satisfied")
            }

            val finalResults = applyNMS(detections)
            val isFinalDanger = finalResults.any { res ->
                val threshold = distanceThresholds[res.label] ?: defaultThreshold
                res.box.height() > threshold
            }

            if (isFinalDanger) {
                triggerSafetyVibration()
            }

            // Update UI on main thread
            handler.post {
                onDetectionUpdate(finalResults, isFinalDanger)
            }

        } catch (e: Exception) {
            Log.e("ObstacleAnalyzer", "Analysis Error: ${e.message}")
        } finally {
            imageProxy.close()
        }
    }

    private fun triggerSafetyVibration() {
        handler.post {
            if (vibrator?.hasVibrator() == true) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    val effect = VibrationEffect.createOneShot(500, 255)
                    vibrator.vibrate(effect)
                } else {
                    @Suppress("DEPRECATION")
                    vibrator.vibrate(500)
                }
            }
        }
    }

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
        val intArea = Math.max(0f, Math.min(b1.right, b2.right) - Math.max(b1.left, b2.left)) *
                      Math.max(0f, Math.min(b1.bottom, b2.bottom) - Math.max(b1.top, b2.top))
        val uArea = (b1.width() * b1.height()) + (b2.width() * b2.height()) - intArea
        return if (uArea > 0) intArea / uArea else 0f
    }
}

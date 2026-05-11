package com.example.ainavigationforblindpeople.data

import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer

data class DetectionResult(
    val label: String,
    val score: Float,
    val box: RectF
)

class DetectionRepository(private val interpreter: Interpreter) {

    /**
     * Runs object detection on the provided image buffer.
     * Expects YOLOv8/v11 format output: [1, 4 + num_classes, 8400]
     */
    fun detectObjects(buffer: ByteBuffer, labels: List<String>): List<DetectionResult> {
        val numClasses = 80 // Default COCO classes
        // Note: If using custom model with more classes, this should be adjusted.
        // Based on labels provided, it seems there are 86 labels (80 COCO + 6 Currency).
        val actualClasses = if (labels.size >= 86) 86 else 80
        val outputChannels = actualClasses + 4
        
        val output = Array(1) { Array(outputChannels) { FloatArray(8400) } }
        
        try {
            interpreter.run(buffer, output)
        } catch (e: Exception) {
            // Fallback if output shape doesn't match
            return emptyList()
        }

        val detections = mutableListOf<DetectionResult>()
        
        for (i in 0 until 8400) {
            var maxScore = 0f
            var classId = -1
            for (c in 0 until actualClasses) {
                val score = output[0][c + 4][i]
                if (score > maxScore) {
                    maxScore = score
                    classId = c
                }
            }

            if (maxScore > 0.45f) { // Confidence threshold
                val label = labels.getOrElse(classId) { "Object" }
                val cx = output[0][0][i]
                val cy = output[0][1][i]
                val w = output[0][2][i]
                val h = output[0][3][i]

                // Convert center coordinates to bounding box
                val box = RectF(
                    (cx - w / 2f) / 640f,
                    (cy - h / 2f) / 640f,
                    (cx + w / 2f) / 640f,
                    (cy + h / 2f) / 640f
                )

                detections.add(DetectionResult(label, maxScore, box))
            }
        }

        return applyNMS(detections)
    }

    private fun applyNMS(results: List<DetectionResult>): List<DetectionResult> {
        val sorted = results.sortedByDescending { it.score }
        val selected = mutableListOf<DetectionResult>()
        for (res in sorted) {
            if (selected.none { calculateIoU(res.box, it.box) > 0.45f }) {
                selected.add(res)
                if (selected.size >= 10) break
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

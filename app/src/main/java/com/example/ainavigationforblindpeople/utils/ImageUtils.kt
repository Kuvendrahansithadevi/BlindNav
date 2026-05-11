package com.example.ainavigationforblindpeople.utils

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.camera.core.ImageProxy

object ImageUtils {

    /**
     * Convert CameraX ImageProxy to Bitmap
     */
    fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            val buffer = imageProxy.planes[0].buffer
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Rotate bitmap by given degrees
     */
    fun rotateBitmap(bitmap: Bitmap, degrees: Int): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(degrees.toFloat())
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    /**
     * Resize bitmap to target dimensions
     */
    fun resizeBitmap(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        return Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
    }

    /**
     * Calculate object size ratio from bounding box
     */
    fun calculateObjectSizeRatio(boxLeft: Float, boxTop: Float, boxRight: Float, boxBottom: Float): Float {
        val width = boxRight - boxLeft
        val height = boxBottom - boxTop
        return width * height
    }

    /**
     * Check if object is vehicle based on label
     */
    fun isVehicle(label: String): Boolean {
        val vehicles = listOf("car", "bus", "truck", "motorcycle", "bicycle", "auto", "vehicle")
        return vehicles.any { label.contains(it, ignoreCase = true) }
    }
}
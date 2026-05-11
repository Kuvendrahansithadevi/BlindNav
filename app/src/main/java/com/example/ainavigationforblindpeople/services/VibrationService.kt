package com.example.ainavigationforblindpeople.services

import android.content.Context
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.VibrationEffect
import android.os.Vibrator
import android.util.Log

class VibrationService(private val context: Context) {
    private var vibrator: Vibrator? = context.getSystemService(Vibrator::class.java)
    private var lastVibrationTime = 0L

    /**
     * Trigger vibration with intensity (0-1 range)
     * Higher intensity = longer duration and stronger amplitude
     */
    fun triggerVibration(intensity: Float = 1.0f) {
        val duration = when {
            intensity >= 0.8f -> 600L
            intensity >= 0.5f -> 300L
            else -> 150L
        }

        val amplitude = (intensity * 255).toInt().coerceIn(0, 255)
        triggerVibration(duration, amplitude)
    }

    /**
     * Trigger vibration with specific duration and amplitude
     */
    fun triggerVibration(duration: Long, amplitude: Int = VibrationEffect.DEFAULT_AMPLITUDE) {
        val v = vibrator
        if (v == null || !v.hasVibrator()) {
            Log.e("VibrationService", "Vibrator not available")
            return
        }

        val currentTime = System.currentTimeMillis()
        if (currentTime - lastVibrationTime < 200) {
            Log.d("VibrationService", "Throttling vibration")
            return
        }
        lastVibrationTime = currentTime

        Handler(Looper.getMainLooper()).post {
            try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    v.vibrate(VibrationEffect.createOneShot(duration, amplitude))
                    Log.d("VibrationService", "Vibrating for ${duration}ms with amplitude $amplitude")
                } else {
                    @Suppress("DEPRECATION")
                    v.vibrate(duration)
                    Log.d("VibrationService", "Vibrating for ${duration}ms (legacy)")
                }
            } catch (e: Exception) {
                Log.e("VibrationService", "Vibration error: ${e.message}")
            }
        }
    }

    /**
     * Stop all ongoing vibrations
     */
    fun cancel() {
        vibrator?.cancel()
        Log.d("VibrationService", "Vibration cancelled")
    }
}
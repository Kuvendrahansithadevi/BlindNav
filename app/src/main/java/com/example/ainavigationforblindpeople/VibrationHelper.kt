package com.example.ainavigationforblindpeople

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log

class VibrationHelper(private val context: Context) {
    private val vibrator: Vibrator? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        val vibratorManager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager
        vibratorManager?.defaultVibrator
    } else {
        @Suppress("DEPRECATION")
        context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
    }

    /**
     * Triggers a vibration for the specified duration.
     * @param duration Duration in milliseconds.
     */
    fun triggerVibration(duration: Long = 500L) {
        val v = vibrator ?: return
        if (!v.hasVibrator()) {
            Log.w("VibrationHelper", "Device does not have a vibrator")
            return
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            v.vibrate(VibrationEffect.createOneShot(duration, VibrationEffect.DEFAULT_AMPLITUDE))
        } else {
            @Suppress("DEPRECATION")
            v.vibrate(duration)
        }
        Log.d("VibrationHelper", "Vibration triggered for ${duration}ms")
    }
}
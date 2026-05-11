package com.example.ainavigationforblindpeople.services

import android.Manifest
import android.content.Context
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.telephony.SmsManager
import android.util.Log
import android.widget.Toast
import androidx.core.app.ActivityCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority

class SOSService(private val context: Context) {
    private val prefs: SharedPreferences = context.getSharedPreferences("BlindNavPrefs", Context.MODE_PRIVATE)
    private lateinit var fusedLocationClient: FusedLocationProviderClient

    /**
     * Initialize location client
     */
    fun initLocationClient() {
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(context)
    }

    /**
     * Save guardian phone number
     */
    fun saveGuardianNumber(number: String) {
        prefs.edit().putString("guardian_no", number).apply()
        Log.d("SOSService", "Guardian number saved: $number")
    }

    /**
     * Get saved guardian phone number
     */
    fun getGuardianNumber(): String {
        return prefs.getString("guardian_no", "") ?: ""
    }

    /**
     * Check if guardian number is saved
     */
    fun hasGuardianNumber(): Boolean {
        return getGuardianNumber().isNotEmpty()
    }

    /**
     * Send SOS with current location
     */
    fun sendSOS(onResult: (Boolean, String) -> Unit) {
        val number = getGuardianNumber()

        if (number.isEmpty()) {
            Log.e("SOSService", "No guardian number saved")
            onResult(false, "No guardian number saved")
            return
        }

        // Check SMS permission
        if (ActivityCompat.checkSelfPermission(context, Manifest.permission.SEND_SMS) != PackageManager.PERMISSION_GRANTED) {
            Log.e("SOSService", "SMS permission not granted")
            onResult(false, "SMS permission not granted")
            return
        }

        // Get current location
        fusedLocationClient.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, null)
            .addOnSuccessListener { location ->
                val message = if (location != null) {
                    "EMERGENCY SOS! Need immediate help. My location: https://maps.google.com/?q=${location.latitude},${location.longitude}"
                } else {
                    "EMERGENCY SOS! Need immediate help. Location unavailable - please call me."
                }
                sendSmsMessage(number, message, onResult)
            }
            .addOnFailureListener { exception ->
                Log.e("SOSService", "Location failed: ${exception.message}")
                sendSmsMessage(number, "EMERGENCY SOS! Need immediate help. Location unavailable - please call me.", onResult)
            }
    }

    private fun sendSmsMessage(phoneNumber: String, message: String, onResult: (Boolean, String) -> Unit) {
        try {
            val smsManager = SmsManager.getDefault()
            smsManager.sendTextMessage(phoneNumber, null, message, null, null)
            Log.w("SOSService", "SMS sent successfully to $phoneNumber")
            onResult(true, "SOS sent successfully")
        } catch (e: Exception) {
            Log.e("SOSService", "Failed to send SMS: ${e.message}")
            onResult(false, "Failed to send SMS: ${e.message}")
        }
    }
}
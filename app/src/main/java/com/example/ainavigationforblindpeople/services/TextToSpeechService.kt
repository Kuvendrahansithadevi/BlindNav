package com.example.ainavigationforblindpeople.services

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeech.OnInitListener
import android.util.Log
import java.util.Locale

class TextToSpeechService(context: Context) : OnInitListener {

    private var tts: TextToSpeech? = TextToSpeech(context, this)
    private var isReady = false
    private var currentLanguage: Locale = Locale.US

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            isReady = true
            tts?.language = currentLanguage
            tts?.setSpeechRate(0.8f)
            Log.d("TTSService", "TextToSpeech initialized successfully")
            speak("App ready")
        } else {
            Log.e("TTSService", "TextToSpeech initialization failed")
        }
    }

    /**
     * Speak text with current language
     */
    fun speak(text: String) {
        if (!isReady) {
            Log.e("TTSService", "TTS not ready")
            return
        }

        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
        Log.d("TTSService", "Speaking: $text")
    }

    /**
     * Change language for speech output
     */
    fun setLanguage(languageCode: String): Boolean {
        val locale = when (languageCode.lowercase()) {
            "telugu" -> Locale("te", "IN")
            "hindi" -> Locale("hi", "IN")
            "tamil" -> Locale("ta", "IN")
            "kannada" -> Locale("kn", "IN")
            "english" -> Locale.US
            else -> Locale.US
        }

        currentLanguage = locale
        tts?.language = locale
        Log.d("TTSService", "Language changed to: $languageCode")
        return true
    }

    /**
     * Get current language
     */
    fun getCurrentLanguage(): String {
        return when (currentLanguage.language) {
            "te" -> "Telugu"
            "hi" -> "Hindi"
            "ta" -> "Tamil"
            "kn" -> "Kannada"
            else -> "English"
        }
    }

    /**
     * Shutdown TTS service
     */
    fun shutdown() {
        tts?.stop()
        tts?.shutdown()
        isReady = false
        Log.d("TTSService", "TTS shutdown")
    }
}
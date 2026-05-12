package com.example.ainavigationforblindpeople

/**
 * Base interface to ensure all future roadmap modules 
 * follow a consistent architecture.
 */
interface BaseFutureFeature {
    fun initializeModule()
    fun getStatus(): String
}

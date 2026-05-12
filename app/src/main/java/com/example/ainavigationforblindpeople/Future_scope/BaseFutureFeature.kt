package com.yourpackage.name.future_scope

/**
 * Base interface to ensure all future roadmap modules 
 * follow a consistent architecture.
 */
interface BaseFutureFeature {
    fun initializeModule()
    fun getStatus(): String
}

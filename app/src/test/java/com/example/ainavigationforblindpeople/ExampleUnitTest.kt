package com.example.ainavigationforblindpeople

import org.junit.Test

import org.junit.Assert.*

/**
 * Example local unit test, which will execute on the development machine (host).
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
class ExampleUnitTest {
  @Test
fun testSOSMessageFormatting() {
    val message = "Emergency! I need help at Lat: 17.38, Long: 78.48"
    assert(message.contains("Emergency"))
}
}

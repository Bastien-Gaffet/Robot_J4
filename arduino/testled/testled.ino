#include <Adafruit_NeoPixel.h>
// This sketch is designed to test a strip of NeoPixels by lighting them up one by one.
// It will stop when it reaches the end of the strip or when an LED does not light up.
#define PIN       6
#define MAX_TEST  120  // Maximum number of LEDs to test (adjust as needed)

Adafruit_NeoPixel strip(MAX_TEST, PIN, NEO_GRB + NEO_KHZ800);

void setup() {
  Serial.begin(9600);
  strip.begin();
  strip.show();
  delay(2000);  // Wait before the test

  int lastWorkingLED = -1;

  for (int i = 0; i < MAX_TEST; i++) {
    strip.clear();  // Turns off all LEDs
    strip.setPixelColor(i, strip.Color(0, 255, 0));  // Tries to light up an LED in green
    strip.show();
    delay(10); // Gives time to see the lighting up

    // Visually check if the LED lights up
    // As soon as no LED lights up, it's probably the end
    Serial.print("Test LED ");
    Serial.println(i);
  }

  // Visually note the last LED that responded
  Serial.println("Test terminé. Note la dernière LED qui s’est allumée.");
}

void loop() {
  // Nothing here
}

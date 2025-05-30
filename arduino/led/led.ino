#include <Adafruit_NeoPixel.h>

#define PIN        2     // Data pin
#define NUMPIXELS 120       // Total number of LEDs

Adafruit_NeoPixel strip(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);

// --- SETUP ---
void setup() {
  strip.begin();
  strip.setBrightness(25);
  strip.show(); // Turns off all LEDs
}

// --- LOOP ---
void loop() {
  ledsReds();
  delay(1000);

  ledsYellows();
  delay(1000);

  arcEnCiel(0);  // Scrolling speed (smaller = faster)
}

// --- FUNCTION : Red ---
void ledsReds() {
  for(int i=0; i<strip.numPixels(); i++) {
    strip.setPixelColor(i, strip.Color(255, 0, 0)); // Red
  }
  strip.show();
}

// --- FUNCTION : Yellow ---
void ledsYellows() {
  for(int i=0; i<strip.numPixels(); i++) {
    strip.setPixelColor(i, strip.Color(255, 255, 0)); // Yellow (R + G)
  }
  strip.show();
}

// --- FUNCTION : Progressive rainbow ---
void arcEnCiel(uint8_t wait) {
  for(long firstPixelHue = 0; firstPixelHue < 50*65536; firstPixelHue += 512) {
    for(int i=0; i<strip.numPixels(); i++) {
      int pixelHue = firstPixelHue + (i * 65536L / strip.numPixels());
      strip.setPixelColor(i, strip.gamma32(strip.ColorHSV(pixelHue)));
    }
    strip.show();
    delay(wait);
  }
}

//program uploaded to GitHub on 05/30/2025
#include <Servo.h>
#include <Adafruit_NeoPixel.h>

//definition of the LED strip
#define PIN        6       //LED pin
#define NUMPIXELS 120       //Total number of LEDs
long firstPixelHue = 0;
Adafruit_NeoPixel strip(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);


//Global declaration of positions for each column for the placerdessuscolonne function
struct PositionColonne {
  long step2;
  long step2b;
  long step3;
  long step1;
  long step5;
};

const PositionColonne positions[] = {
  { -300,  300, -7000, -1700, -1520 }, // column 1
  { -300,  300, -7000, -1700, -1520 }, // column 2
  { -300,  300, -7000, -1700, -1520 }, // column 3
  { -300,  300, -7000, -1700, -1520 }, // column 4
  { -300,  300, -7000, -1700, -1520 }, // column 5
  { -800,  800, -5000, -1830, -1520 }, // column 6
  { -960,  960, -4000, -1830, -1520 }, // column 7
};

const int servPin=8; // PIN servomotor
Servo servo_Pince;
const int Maxspeed1=5000.0;     //definition of the maximum speed of stepper 1
const int Accel1=1000.0;        //definition of the acceleration of stepper 1
const int Maxspeed2=2500.0;
const int Accel2=500.0;
const int Maxspeed3=5000.0;
const int Accel3=1000.0;
const int Maxspeed4=5000.0;
const int Accel4=1000.0;
const int Maxspeed5=5000.0;
const int Accel5=1000.0;
int entree;
bool premierPrendreJeton = true;  //Flag to detect the first call to prendreJeton()

#include <AccelStepper.h>     //include the AccelStepper library
// Driver 1 
const int dirPin_1=35;        //definition of the wiring for Stepper1
const int stepPin_1=37;
// Driver 2
const int dirPin_2=22;
const int stepPin_2=24;
// Driver 2b
const int dirPin_2b=25;
const int stepPin_2b=27;
// Driver 3
const int dirPin_3=29;
const int stepPin_3=31;
// Driver 4
const int dirPin_4=41;
const int stepPin_4=43;
// Driver 5
const int dirPin_5=51;
const int stepPin_5=53;
// Driver substep
const int subdiv=8;
//definition of the open and closed positions of the servomotor
int servof = 40;
int servoo = 0;
//Position value in steps
//Create an instance
AccelStepper stepper_1(AccelStepper::DRIVER,stepPin_1,dirPin_1);          //creation of the stepper_1 subroutines using the AccelStepper library
AccelStepper stepper_2(AccelStepper::DRIVER,stepPin_2,dirPin_2);
AccelStepper stepper_2b(AccelStepper::DRIVER,stepPin_2b,dirPin_2b);
AccelStepper stepper_3(AccelStepper::DRIVER,stepPin_3,dirPin_3);
AccelStepper stepper_4(AccelStepper::DRIVER,stepPin_4,dirPin_4);
AccelStepper stepper_5(AccelStepper::DRIVER,stepPin_5,dirPin_5);

void prendreJeton(int pas1, int pas2, int pas3, int pas4, int pas5);

void setup(){
  strip.begin();
  strip.setBrightness(150);
  strip.show(); //Turns off all the LEDs
  servo_Pince.attach(servPin);        //initialization of the servo motor connection for the gripper
  servo_Pince.write(servoo);
  delay(200);
  servo_Pince.write(servof);
  delay(200);
  servo_Pince.write(servoo);
  delay(200);
  servo_Pince.write(servof);

  //Driver connections
  // pinMode(bpPin,INPUT);
  //******** 
  //Speed and acceleration configuration
  stepper_1.setMaxSpeed(Maxspeed1); //max speed i, steps per second
  stepper_1.setAcceleration(Accel1); //acceleration in steps per second²
  stepper_2.setMaxSpeed(Maxspeed2); //max speed i, steps per second
  stepper_2.setAcceleration(Accel2); //acceleration in steps per second²
  stepper_2b.setMaxSpeed(Maxspeed2); //max speed i, steps per second
  stepper_2b.setAcceleration(Accel2); //acceleration in steps per second²
  stepper_3.setMaxSpeed(Maxspeed3); //max speed i, steps per second
  stepper_3.setAcceleration(Accel3); //acceleration in steps per second²
  stepper_4.setMaxSpeed(Maxspeed4); //max speed i, steps per second
  stepper_4.setAcceleration(Accel4); //acceleration in steps per second²
  stepper_5.setMaxSpeed(Maxspeed5); //max speed i, steps per second
  stepper_5.setAcceleration(Accel5); //acceleration in steps per second²
  Serial.begin(9600);
  Serial.setTimeout(1000);
  prendreJeton(0, 1480, 680, 800, 900);
}

// --- FUNCTION: Red ---
void ledsRouges() {
  for(int i=0; i<strip.numPixels(); i++) {
    strip.setPixelColor(i, strip.Color(255, 0, 0)); // Red
  }
  strip.show();
}

void flashRouge(int duree_on, int duree_off, int nb_flash) {
  for (int i = 0; i < nb_flash; i++) {
    // Turn on in red
    for (int j = 0; j < strip.numPixels(); j++) {
      strip.setPixelColor(j, strip.Color(255, 0, 0));
    }
    strip.show();
    delay(duree_on);

    // Turn off
    for (int j = 0; j < strip.numPixels(); j++) {
      strip.setPixelColor(j, strip.Color(0, 0, 0));
    }
    strip.show();
    delay(duree_off);
  }
  ledsRouges();
}

// --- FUNCTION : Orange ---
void ledsOrange() {
  for(int i=0; i<strip.numPixels(); i++) {
    strip.setPixelColor(i, strip.Color(255, 70, 0)); // Orange
  }
  strip.show();
}

void flashOrange(int duree_on, int duree_off, int nb_flash) {
  for (int i = 0; i < nb_flash; i++) {
    // Turn on in orange
    for (int j = 0; j < strip.numPixels(); j++) {
      strip.setPixelColor(j, strip.Color(255, 70, 0));
    }
    strip.show();
    delay(duree_on);

    // Turn off
    for (int j = 0; j < strip.numPixels(); j++) {
      strip.setPixelColor(j, strip.Color(0, 0, 0));
    }
    strip.show();
    delay(duree_off);
  }
  ledsOrange();
}

// --- FUNCTION : Yellow ---
void ledsJaunes() {
  for(int i=0; i<strip.numPixels(); i++) {
    strip.setPixelColor(i, strip.Color(255, 155, 0)); // Yellow (R + G)
  }
  strip.show();
}

void flashJaune(int duree_on, int duree_off, int nb_flash) {
  for (int i = 0; i < nb_flash; i++) {
    // Turn on in yellow
    for (int j = 0; j < strip.numPixels(); j++) {
      strip.setPixelColor(j, strip.Color(255, 155, 0));
    }
    strip.show();
    delay(duree_on);

    // Turn off
    for (int j = 0; j < strip.numPixels(); j++) {
      strip.setPixelColor(j, strip.Color(0, 0, 0));
    }
    strip.show();
    delay(duree_off);
  }
  ledsJaunes();
}

void placerDessusColonne(int colonne) {
  if (colonne < 1 || colonne > 7) return;

  const PositionColonne& pos = positions[colonne - 1];

  // Phase 1: Start steppers 1, 3, and 5
  stepper_1.moveTo(pos.step1);
  stepper_3.moveTo(pos.step3);
  stepper_5.moveTo(pos.step5);

  // We calculate the absolute distances for each stepper
  long dist1 = abs(stepper_1.distanceToGo());
  long dist3 = abs(stepper_3.distanceToGo());
  long dist5 = abs(stepper_5.distanceToGo());

  // Define a trigger threshold (for example, 20% of the travel)
  long seuil1 = dist1 * 0.2;
  long seuil3 = dist3 * 0.2;
  long seuil5 = dist5 * 0.2;

  // Initial loop: run until reaching 20% of the travel distance
  while (abs(stepper_1.distanceToGo()) > seuil1 ||
         abs(stepper_3.distanceToGo()) > seuil3 ||
         abs(stepper_5.distanceToGo()) > seuil5) {
    stepper_1.run();
    stepper_3.run();
    stepper_5.run();
  }

  // Phase 2: start steppers 2 and 2b
  stepper_2.moveTo(pos.step2);
  stepper_2b.moveTo(pos.step2b);

  // Final loop: everyone until the end
  while (stepper_1.distanceToGo() != 0 ||
         stepper_2.distanceToGo() != 0 ||
         stepper_2b.distanceToGo() != 0 ||
         stepper_3.distanceToGo() != 0 ||
         stepper_5.distanceToGo() != 0) {
    stepper_1.run();
    stepper_2.run();
    stepper_2b.run();
    stepper_3.run();
    stepper_5.run();
  }
}


void posmvt(){
  // The motors move to an optimized position for movement above the column
  stepper_2.moveTo(-300);
  stepper_2b.moveTo(300);
  stepper_3.moveTo(-700*10);
  
  // First execute steppers 2, 2b, and 3 up to a certain point (for example, 20% of the travel)
  int initialDistance2 = stepper_2.distanceToGo();
  int initialDistance2b = stepper_2b.distanceToGo();
  int initialDistance3 = stepper_3.distanceToGo();
  
  // Calculate the point (20% of the travel) where steppers 1 and 5 will start
  int threshold2 = abs(initialDistance2) * 0.2;
  int threshold2b = abs(initialDistance2b) * 0.2;
  int threshold3 = abs(initialDistance3) * 0.2;
  
  // First phase: only steppers 2, 2b, and 3
  while ((abs(stepper_2.distanceToGo()) > threshold2) || (abs(stepper_2b.distanceToGo()) > threshold2b) || (abs(stepper_3.distanceToGo()) > threshold3)) {
    stepper_2.run();
    stepper_2b.run();
    stepper_3.run();
  }
  
  // Now, start steppers 1 and 5 for the second phase
  stepper_1.moveTo(-1700);
  stepper_5.moveTo(-1520);
  
  // Second phase: all steppers together
  while (stepper_1.distanceToGo() != 0 || stepper_2.distanceToGo() != 0 || stepper_2b.distanceToGo() != 0 || stepper_3.distanceToGo() != 0 || stepper_5.distanceToGo() != 0) {
    stepper_1.run();
    stepper_2.run();
    stepper_2b.run();
    stepper_3.run();
    stepper_5.run();
  }
}   

void prendreJeton(int pas1, int pas2, int pas3, int pas4, int pas5){// Definition of the number of steps to execute (for now, this number of steps is common to all steppers)

  if (premierPrendreJeton) {
    servo_Pince.write(0);
    stepper_2.moveTo(pas2*-1);
    stepper_2b.moveTo(pas2);                      // The step count for motor 2b must be the opposite of motor 2
    stepper_3.moveTo(-pas3*10);
    stepper_5.moveTo(pas5*-1);

    while (stepper_3.distanceToGo()!=0||stepper_5.distanceToGo()!=0){
        stepper_3.run();
        stepper_5.run();
    }
    
    while (stepper_2.distanceToGo()!=0||stepper_2b.distanceToGo()!=0){
      stepper_2.run();
      stepper_2b.run();
    }

    delay(200);
    servo_Pince.write(servof);
    delay(200);
    premierPrendreJeton = false;
  } 
  
  else {
    servo_Pince.write(0);
    stepper_2.moveTo(pas2*-1);
    stepper_2b.moveTo(pas2);
    stepper_3.moveTo(-pas3*10);
    stepper_5.moveTo(pas5*-1);


    while (stepper_2.distanceToGo()!=0||stepper_2b.distanceToGo()!=0||stepper_3.distanceToGo()!=0||stepper_5.distanceToGo()!=0){
      stepper_2.run();
      stepper_2b.run();
      stepper_3.run();
      stepper_5.run();
    }
    
    delay(200);
    servo_Pince.write(servof);
    delay(200);
  }
}

void poserJeton(int pas1, int pas2, int pas3, int pas4, int pas5){
  servo_Pince.write(servof);
  stepper_1.moveTo(-pas1);                       
  stepper_2.moveTo(-pas2);
  stepper_2b.moveTo(pas2);                     
  stepper_3.moveTo(-pas3*10);                    
  stepper_5.moveTo(pas5*-1);

  while(stepper_1.distanceToGo()!=0){  
    stepper_1.run();
  }
    
  while(stepper_2.distanceToGo()!=0||stepper_2b.distanceToGo()!=0||stepper_3.distanceToGo()!=0||stepper_5.distanceToGo()!=0){  
    stepper_2.run();
    stepper_2b.run();
    stepper_3.run();
    stepper_5.run();
  }
  
  servo_Pince.write(servoo);
  delay (200);
  posdistributeur(pas1, pas2, pas3, pas4, pas5);
  prendreJeton(0, 1480, 680, 800, 900);
}

void posdistributeur(int pas1, int pas2, int pas3, int pas4, int pas5) {
  //position to place the robot above the dispenser depending on the position on the grid
  stepper_1.moveTo(0);                           
  stepper_2.moveTo(-pas2+100);
  stepper_2b.moveTo(pas2-100);                      
  stepper_3.moveTo(-pas3*10+1500);                      
  stepper_5.moveTo(-pas5-100);

  int initialDistance2 = stepper_2.distanceToGo();
  int initialDistance2b = stepper_2b.distanceToGo();
  int initialDistance3 = stepper_3.distanceToGo();
  
  int threshold2 = abs(initialDistance2) * 0.5;
  int threshold2b = abs(initialDistance2b) * 0.5;
  int threshold3 = abs(initialDistance3) * 0.5;
  
  // First phase: only steppers 2, 2b, and 3
  while ((abs(stepper_2.distanceToGo()) > threshold2) || (abs(stepper_2b.distanceToGo()) > threshold2b) || (abs(stepper_3.distanceToGo()) > threshold3)) {
    stepper_2.run();
    stepper_2b.run();
    stepper_3.run();
  }

  // Second phase: all steppers together
  while (stepper_1.distanceToGo() != 0 || stepper_2.distanceToGo() != 0 || stepper_2b.distanceToGo() != 0 || stepper_3.distanceToGo() != 0 || stepper_5.distanceToGo() != 0) {
    stepper_1.run();
    stepper_2.run();
    stepper_2b.run();
    stepper_3.run();
    stepper_5.run();
  }
}

void poserDansColonne(int colonne) {
  placerDessusColonne(colonne);
  if(colonne == 1)
    poserJeton(1700, 140, 920, 0, 1890);
  else if (colonne == 2)
    poserJeton(1730, 50, 950, 0, 1520);
  else if (colonne == 3)
    poserJeton(1780, 250, 860, 0, 1520);
  else if (colonne == 4)
    poserJeton(1800, 460, 750, 0, 1520);
  else if (colonne == 5)
    poserJeton(1800, 620, 650, 0, 1520);
  else if (colonne == 6)
    poserJeton(1830, 830, 520, 0, 1520);
  else if (colonne == 7)
    poserJeton(1850, 990, 420, 0, 1500);
}

void findepartie(int pas1, int pas2, int pas3, int pas4, int pas5) {
  stepper_1.moveTo(0);
  stepper_2.moveTo(pas2*-1);
  stepper_2b.moveTo(pas2);
  stepper_3.moveTo(-pas3*10);
  stepper_5.moveTo(pas5*-1);

  while (stepper_1.distanceToGo()!=0||stepper_2.distanceToGo()!=0||stepper_2b.distanceToGo()!=0||stepper_3.distanceToGo()!=0||stepper_5.distanceToGo()!=0){
    stepper_1.run();
    stepper_2.run();
    stepper_2b.run();
    stepper_3.run();
    stepper_5.run();
  }
  servo_Pince.write(servoo);
  posdistributeur(1780, 250, 860, 0, 1520);
}

void victoireRouge()  {
  flashRouge(50, 50, 5);
  findepartie(0, 470, 1100, 800, 950);
}

void victoireJaune()  {
  flashJaune(50, 50, 5);
  findepartie(0, 470, 1100, 800, 950);
}

void matchNul() {
  flashOrange(50, 50, 5);
  findepartie(0, 470, 1100, 800, 950);
}

void loop() {
    if (Serial.available()) {
        String teststr = Serial.readStringUntil('\n');  // or '\r' depending on your system
        teststr.trim();
        int entree = teststr.toInt();

        if (entree >= 1 && entree <= 7) {           
            poserDansColonne(entree);
        } else {
            switch (entree) {
                case 8: ledsRouges(); break;
                case 9: ledsJaunes(); break;
                case 12: posmvt(); break;
                case 20: matchNul(); break;
                case 21: victoireRouge(); break;
                case 22: victoireJaune(); break;
            }
        }
    }
}

#include <Servo.h>
#include <AccelStepper.h>

// Declaration of pins and servomotors
const int servPin = 8; // PIN for gripper servomotor
Servo servo_Pince;

// Stepper motor configurations
const int dirPin_1 = 35;
const int stepPin_1 = 37;
const int dirPin_2 = 22;
const int stepPin_2 = 24;
const int dirPin_2b = 25;
const int stepPin_2b = 27;
const int dirPin_3 = 29;
const int stepPin_3 = 31;
const int dirPin_4 = 41;
const int stepPin_4 = 43;
const int dirPin_5 = 51;
const int stepPin_5 = 53;

// Creating instances of stepper motors
AccelStepper stepper_1(AccelStepper::DRIVER, stepPin_1, dirPin_1);
AccelStepper stepper_2(AccelStepper::DRIVER, stepPin_2, dirPin_2);
AccelStepper stepper_2b(AccelStepper::DRIVER, stepPin_2b, dirPin_2b);
AccelStepper stepper_3(AccelStepper::DRIVER, stepPin_3, dirPin_3);
AccelStepper stepper_4(AccelStepper::DRIVER, stepPin_4, dirPin_4);
AccelStepper stepper_5(AccelStepper::DRIVER, stepPin_5, dirPin_5);

// Configuration variables
const int Maxspeed1 = 5000.0;
const int Accel1 = 1000.0;
const int servof = 40;
const int servoo = 0;

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(1000);
  
  // Servo configuration
  servo_Pince.attach(servPin);
  servo_Pince.write(servoo);
  
  // Stepper motors speed configuration
  stepper_1.setMaxSpeed(Maxspeed1);
  stepper_1.setAcceleration(Accel1);
  stepper_2.setMaxSpeed(2500.0);
  stepper_2.setAcceleration(500.0);
  stepper_2b.setMaxSpeed(2500.0);
  stepper_2b.setAcceleration(500.0);
  stepper_3.setMaxSpeed(5000.0);
  stepper_3.setAcceleration(1000.0);
  stepper_4.setMaxSpeed(5000.0);
  stepper_4.setAcceleration(1000.0);
  stepper_5.setMaxSpeed(5000.0);
  stepper_5.setAcceleration(1000.0);

  Serial.println("Robot Control System Ready");
  Serial.println("Commands:");
  Serial.println("- 'pos': Get current positions");
  Serial.println("- 'move [stepper] [steps]': Move a specific stepper");
  Serial.println("- 'servo [angle]': Move servo");
  Serial.println("- 'open': Ouvrir la pince");
  Serial.println("- 'close': Fermer la pince");
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command == "pos") {
      getPositions();
    }
    else if (command.startsWith("move")) {
      moveStepper(command);
    }
    else if (command.startsWith("servo")) {
      moveServo(command);
    }
    else if (command == "open" || command == "close") {
      controlPince(command);
    }
  }
  
  // Execution of ongoing movements
  stepper_1.run();
  stepper_2.run();
  stepper_2b.run();
  stepper_3.run();
  stepper_4.run();
  stepper_5.run();
}

void getPositions() {
  Serial.println("Current Positions:");
  Serial.print("Stepper 1: "); Serial.println(stepper_1.currentPosition());
  Serial.print("Stepper 2: "); Serial.println(stepper_2.currentPosition());
  Serial.print("Stepper 2b: "); Serial.println(stepper_2b.currentPosition());
  Serial.print("Stepper 3: "); Serial.println(stepper_3.currentPosition());
  Serial.print("Stepper 4: "); Serial.println(stepper_4.currentPosition());
  Serial.print("Stepper 5: "); Serial.println(stepper_5.currentPosition());
  Serial.print("Servo: "); Serial.println(servo_Pince.read());
}

void moveStepper(String command) {
  // Expected format: "move [stepper] [steps]"
  int firstSpace = command.indexOf(' ');
  int secondSpace = command.indexOf(' ', firstSpace + 1);
  
  if (firstSpace == -1 || secondSpace == -1) {
    Serial.println("Invalid move command. Use 'move [stepper] [steps]'");
    return;
  }
  
  String stepperStr = command.substring(firstSpace + 1, secondSpace);
  long steps = command.substring(secondSpace + 1).toInt();
  
  if (stepperStr == "1") {
    stepper_1.moveTo(steps);
  } else if (stepperStr == "2") {
    // Move stepper 2 and 2b simultaneously with opposite steps
    stepper_2.moveTo(steps);
    stepper_2b.moveTo(-steps);
  } else if (stepperStr == "3") {
    stepper_3.moveTo(steps);
  } else if (stepperStr == "4") {
    stepper_4.moveTo(steps);
  } else if (stepperStr == "5") {
    stepper_5.moveTo(steps);
  } else {
    Serial.println("Invalid stepper number");
  }
}

void moveServo(String command) {
  // Expected format: "servo [angle]"
  int firstSpace = command.indexOf(' ');
  
  if (firstSpace == -1) {
    Serial.println("Invalid servo command. Use 'servo [angle]'");
    return;
  }
  
  int angle = command.substring(firstSpace + 1).toInt();
  
  if (angle >= 0 && angle <= 180) {
    servo_Pince.write(angle);
    Serial.print("Servo moved to: ");
    Serial.println(angle);
  } else {
    Serial.println("Angle must be between 0 and 180");
  }
}

void controlPince(String command) {
  if (command == "open") {
    // Fully open position (0 degrees)
    servo_Pince.write(servoo);
    Serial.println("Pince ouverte");
  } else if (command == "close") {
    // Fully closed position (40 degrees or value of servof)
    servo_Pince.write(servof);
    Serial.println("Pince fermée");
  } else {
    Serial.println("Commandes valides : 'open' ou 'close'");
  }
}

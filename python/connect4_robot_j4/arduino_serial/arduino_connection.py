import serial
import serial.tools.list_ports

def detect_arduino():
    """
    Detect Arduino port with improved error handling and multiple identification methods.

    Returns:
    - Port name if Arduino is found
    - None if no Arduino is detected
    """
    ports = list(serial.tools.list_ports.comports())

    # Extended port identification methods
    arduino_keywords = [
        "Arduino", "CH340", "USB Serial",
        "Silicon Labs", "CP210x", "FTDI"
    ]

    for port in ports:
        # Check for keywords in description or hardware ID
        for keyword in arduino_keywords:
            if (keyword.lower() in str(port.description).lower() or
                keyword.lower() in str(port.hwid).lower()):
                return port.device

    return None

def setup_arduino_connection():
    """
    Set up Arduino connection with robust error handling.

    Returns:
    - SerialObj if connection successful
    - None if connection fails
    """
    arduino_port = detect_arduino()

    if not arduino_port:
        print("No Arduino detected.")
        return None

    try:
        SerialObj = serial.Serial(
            port=arduino_port,
            baudrate=9600,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=1
        )
        print(f"Arduino connected successfully on {arduino_port}")
        return SerialObj

    except serial.SerialException:
        # Silently fail and return None without error messages
        return None
    except Exception:
        # Silently fail for other exceptions too
        return None

def send_to_arduino(serial_obj, message):
    #Envoie un message à l'Arduino via la connexion série.
    #Args:
    #    serial_obj (serial.Serial or None): Objet série
    #    message (str or int): Message à envoyer (ex: '12' ou 12)

    if serial_obj is not None:
        try:
            serial_obj.write(f"{message}\n".encode())
            print(f"[Arduino] Sent: {message}")
        except Exception as e:
            print(f"[Arduino] Failed to send '{message}': {e}")
    else:
        print(f"[Arduino] Not connected. Message not sent: {message}")
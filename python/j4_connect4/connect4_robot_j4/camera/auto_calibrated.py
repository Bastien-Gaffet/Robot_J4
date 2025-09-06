import cv2
import numpy as np
from connect4_robot_j4 import constants as cs
import time

class AutoHSVCalibrator:
    def __init__(self):
        # Seuils de référence (vos valeurs actuelles)
        self.reference_ranges = {
            'red1': {'lower': np.array([0, 50, 50]), 'upper': np.array([10, 255, 255])},
            'red2': {'lower': np.array([170, 50, 50]), 'upper': np.array([180, 255, 255])},
            'yellow1': {'lower': np.array([15, 100, 100]), 'upper': np.array([30, 255, 255])},
            'yellow2': {'lower': np.array([30, 100, 100]), 'upper': np.array([45, 255, 255])},
            'yellow3': {'lower': np.array([26, 140, 200]), 'upper': np.array([32, 255, 255])},
            'yellow4': {'lower': np.array([25, 130, 200]), 'upper': np.array([33, 255, 255])}
        }
        
        # Seuils ajustés (seront modifiés automatiquement)
        self.adjusted_ranges = {}
        self.reset_to_reference()
        
        # Paramètres d'ajustement automatique
        self.brightness_offset = 0    # Ajustement V (luminosité)
        self.saturation_offset = 0    # Ajustement S (saturation)
        self.hue_tolerance = 0        # Extension H (teinte)
        
        # Statistiques pour l'auto-ajustement
        self.detection_history = []
        self.last_auto_adjust = 0
        
    def reset_to_reference(self):
        """Remettre les seuils ajustés aux valeurs de référence"""
        self.adjusted_ranges = {}
        for color, ranges in self.reference_ranges.items():
            self.adjusted_ranges[color] = {
                'lower': ranges['lower'].copy(),
                'upper': ranges['upper'].copy()
            }
    
    def setup_trackbars(self):
        """Interface simple avec ajustements globaux"""
        cv2.namedWindow('Auto Calibration', cv2.WINDOW_NORMAL)
        
        # Ajustements globaux
        cv2.createTrackbar('Brightness', 'Auto Calibration', 50, 100, lambda x: None)  # -50 à +50
        cv2.createTrackbar('Saturation', 'Auto Calibration', 50, 100, lambda x: None)  # -50 à +50
        cv2.createTrackbar('Hue Tolerance', 'Auto Calibration', 0, 20, lambda x: None)  # 0 à +20
        
        # Mode automatique
        cv2.createTrackbar('Auto Mode', 'Auto Calibration', 1, 1, lambda x: None)
        cv2.createTrackbar('Show Masks', 'Auto Calibration', 0, 1, lambda x: None)
        
    def get_adjustments(self):
        """Récupérer les ajustements depuis les trackbars"""
        brightness = cv2.getTrackbarPos('Brightness', 'Auto Calibration') - 50  # -50 à +50
        saturation = cv2.getTrackbarPos('Saturation', 'Auto Calibration') - 50   # -50 à +50
        hue_tolerance = cv2.getTrackbarPos('Hue Tolerance', 'Auto Calibration')   # 0 à +20
        
        return brightness, saturation, hue_tolerance
        
    def apply_global_adjustments(self, brightness, saturation, hue_tolerance):
        """Appliquer les ajustements à tous les seuils"""
        for color, ref_ranges in self.reference_ranges.items():
            lower = ref_ranges['lower'].copy()
            upper = ref_ranges['upper'].copy()
            
            # Ajuster la luminosité (V)
            lower[2] = np.clip(lower[2] + brightness, 0, 255)
            # upper[2] reste généralement à 255
            
            # Ajuster la saturation (S) 
            lower[1] = np.clip(lower[1] + saturation, 0, 255)
            # upper[1] reste généralement à 255
            
            # Étendre la tolérance de teinte (H)
            if hue_tolerance > 0:
                # Pour les rouges (qui peuvent être sur 0 et 180)
                if 'red' in color:
                    if lower[0] < 90:  # Rouge bas (0-10)
                        lower[0] = max(0, lower[0] - hue_tolerance)
                        upper[0] = min(180, upper[0] + hue_tolerance)
                    else:  # Rouge haut (170-180)
                        lower[0] = max(0, lower[0] - hue_tolerance)
                        upper[0] = min(180, upper[0] + hue_tolerance)
                else:  # Jaunes
                    lower[0] = max(0, lower[0] - hue_tolerance)
                    upper[0] = min(180, upper[0] + hue_tolerance)
            
            self.adjusted_ranges[color] = {'lower': lower, 'upper': upper}
    
    def detect_all_tokens(self, frame):
        """Détecter tous les tokens avec les seuils ajustés"""
        roi = frame[cs.ROI_Y:cs.ROI_Y + cs.ROI_H, cs.ROI_X:cs.ROI_X + cs.ROI_W]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        all_detections = {'red': [], 'yellow': []}
        all_masks = {}
        
        # Détecter chaque type de couleur
        for color_key, ranges in self.adjusted_ranges.items():
            mask = cv2.inRange(hsv, ranges['lower'], ranges['upper'])
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cs.KERNEL)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cs.KERNEL)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            circles = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if cs.MIN_AREA <= area <= cs.MAX_AREA:
                    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                    circularity = area / (np.pi * (radius ** 2))
                    if circularity >= cs.MIN_CIRCULARITY:
                        circles.append((int(cx), int(cy), int(radius)))
            
            # Classer par couleur principale
            color_type = 'red' if 'red' in color_key else 'yellow'
            all_detections[color_type].extend(circles)
            all_masks[color_key] = mask
            
        return all_detections, all_masks
    
    def auto_adjust_lighting(self, frame):
        """Ajustement automatique basé sur l'éclairage de la ROI"""
        roi = frame[cs.ROI_Y:cs.ROI_Y + cs.ROI_H, cs.ROI_X:cs.ROI_X + cs.ROI_W]
        
        # Analyser l'éclairage moyen
        mean_brightness = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        
        # Ajustement automatique basé sur la luminosité
        if mean_brightness < 80:  # Sombre
            brightness_adj = 20
            saturation_adj = -10
        elif mean_brightness > 180:  # Très lumineux
            brightness_adj = -30
            saturation_adj = 10
        else:  # Normal
            brightness_adj = 0
            saturation_adj = 0
            
        return brightness_adj, saturation_adj, 5  # Tolérance de teinte fixe
    
    def run(self, camera_index=0):
        """Lancer la calibration automatique"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Erreur: Impossible d'ouvrir la caméra")
            return
            
        self.setup_trackbars()
        
        print("=== Calibration HSV Automatique ===")
        print("Mode Manuel: Utilisez les sliders")
        print("Mode Auto: Ajustement automatique selon l'éclairage")
        print("'R' - Reset aux valeurs de référence")
        print("'S' - Sauvegarder tous les seuils")
        print("'Q' - Quitter")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            auto_mode = cv2.getTrackbarPos('Auto Mode', 'Auto Calibration') == 1
            show_masks = cv2.getTrackbarPos('Show Masks', 'Auto Calibration') == 1
            
            # Ajustement automatique ou manuel
            if auto_mode and time.time() - self.last_auto_adjust > 1.0:
                brightness, saturation, hue_tolerance = self.auto_adjust_lighting(frame)
                self.last_auto_adjust = time.time()
            else:
                brightness, saturation, hue_tolerance = self.get_adjustments()
            
            # Appliquer les ajustements
            self.apply_global_adjustments(brightness, saturation, hue_tolerance)
            
            # Détecter les tokens
            detections, masks = self.detect_all_tokens(frame)
            
            # Créer l'image de résultat
            result = frame.copy()
            
            # Dessiner la ROI
            cv2.rectangle(result, (cs.ROI_X, cs.ROI_Y), 
                         (cs.ROI_X + cs.ROI_W, cs.ROI_Y + cs.ROI_H), 
                         (0, 255, 0), 2)
            
            # Dessiner les détections
            total_detections = 0
            for color_type, circles in detections.items():
                color_bgr = (0, 0, 255) if color_type == 'red' else (0, 255, 255)
                
                for x, y, r in circles:
                    cv2.circle(result, (cs.ROI_X + x, cs.ROI_Y + y), r, color_bgr, 3)
                    cv2.putText(result, color_type[0].upper(), 
                               (cs.ROI_X + x - 10, cs.ROI_Y + y + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
                
                total_detections += len(circles)
            
            # Afficher les informations
            mode_text = "AUTO" if auto_mode else "MANUAL"
            info_text = f"Mode: {mode_text} | Detections: R={len(detections['red'])} Y={len(detections['yellow'])}"
            cv2.putText(result, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Afficher les ajustements actuels
            adj_text = f"Adjustments: Brightness={brightness:+d} Saturation={saturation:+d} Hue_tol={hue_tolerance}"
            cv2.putText(result, adj_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Camera + Auto Detection', result)
            
            # Afficher les masques si demandé
            if show_masks:
                combined_mask = np.zeros_like(list(masks.values())[0])
                for mask in masks.values():
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
                cv2.imshow('Combined Masks', combined_mask)
            
            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset_to_reference()
                print("Reset aux valeurs de référence")
            elif key == ord('s'):
                self.save_all_ranges()
                
        cap.release()
        cv2.destroyAllWindows()
    
    def save_all_ranges(self):
        """Sauvegarder TOUS les seuils ajustés dans le format original"""
        with open('auto_calibrated_constants.py', 'w') as f:
            f.write("import numpy as np\n\n")
            f.write("# Seuils HSV auto-calibrés pour la détection des couleurs\n")
            f.write("# Générés automatiquement - Gardent la structure originale\n\n")
            
            # Sauvegarder dans le format exact de votre constants.py
            ranges = self.adjusted_ranges
            
            f.write("# Rouge - Plage 1\n")
            f.write(f"LOWER_RED1 = np.array({list(ranges['red1']['lower'])})\n")
            f.write(f"UPPER_RED1 = np.array({list(ranges['red1']['upper'])})\n\n")
            
            f.write("# Rouge - Plage 2  \n")
            f.write(f"LOWER_RED2 = np.array({list(ranges['red2']['lower'])})\n")
            f.write(f"UPPER_RED2 = np.array({list(ranges['red2']['upper'])})\n\n")
            
            f.write("# Jaune - Plage 1\n")
            f.write(f"LOWER_YELLOW1 = np.array({list(ranges['yellow1']['lower'])})\n")
            f.write(f"UPPER_YELLOW1 = np.array({list(ranges['yellow1']['upper'])})\n\n")
            
            f.write("# Jaune - Plage 2\n")
            f.write(f"LOWER_YELLOW2 = np.array({list(ranges['yellow2']['lower'])})\n")
            f.write(f"UPPER_YELLOW2 = np.array({list(ranges['yellow2']['upper'])})\n\n")
            
            f.write("# Jaune - Plage 3\n")
            f.write(f"LOWER_YELLOW3 = np.array({list(ranges['yellow3']['lower'])})\n")
            f.write(f"UPPER_YELLOW3 = np.array({list(ranges['yellow3']['upper'])})\n\n")
            
            f.write("# Jaune - Plage 4\n")
            f.write(f"LOWER_YELLOW4 = np.array({list(ranges['yellow4']['lower'])})\n")
            f.write(f"UPPER_YELLOW4 = np.array({list(ranges['yellow4']['upper'])})\n")
            
        print("TOUS les seuils sauvegardés dans 'auto_calibrated_constants.py'")
        print("Copiez le contenu dans votre constants.py")
        
        # Afficher un résumé des ajustements
        print("\n=== Résumé des seuils ajustés ===")
        for color_key, ranges in self.adjusted_ranges.items():
            print(f"{color_key.upper()}: Lower={list(ranges['lower'])} Upper={list(ranges['upper'])}")

def main():
    calibrator = AutoHSVCalibrator()
    calibrator.run(camera_index=0)

if __name__ == "__main__":
    main()
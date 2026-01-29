import cv2
import numpy as np

class LungSegmenter:
    """
    Segment lungs from Chest X-Rays using OpenCV image processing techniques.
    Allows masking out irrelevant artifacts (tags, wires, etc).
    """
    def __init__(self):
        pass

    def segment(self, image_rgb):
        """
        Segment lungs.
        Input: RGB numpy array (H, W, 3) - 0-255
        Output: Masked Image (H, W, 3), Binary Mask (H, W)
        """
        # 1. Convert to grayscale
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # 2. Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Thresholding
        # Lungs are dark (air), so we invert first to make them white if we simple thresh
        # Actually standard OTSU works on bimodal.
        # Xray: Lungs = Dark, Bones/Tissue = Light.
        # We want to keep Dark. 
        # Binary Inverse Threshold: Dark pixels become White (255), Light become Black (0).
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 4. Morphological Operations (Cleaning)
        # Remove small noise (Erosion) then dilate
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Dilate slightly to capture full lung field
        mask = cv2.dilate(opening, kernel, iterations=1)
        
        # 5. Keep largest connected components (Lungs)
        # Usually 2 largest contours are lungs. But sometimes body is one big contour.
        # This heuristic can be tricky. Let's filter by area.
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image_rgb, np.ones_like(gray) * 255 # Fallback: Return original
            
        # Create an empty mask
        final_mask = np.zeros_like(gray)
        
        # Sort contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Heuristic: Keep large contours that are not the entire image
        height, width = gray.shape
        image_area = height * width
        
        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter huge chunks (likely background if inverted incorrectly) or tiny specks
            if area < 0.05 * image_area: 
                continue # Too small
            if area > 0.9 * image_area:
                continue # Too big (probably the border)
                
            cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            count += 1
            if count >= 2: # Keep top 2 (Left/Right Lung) - or 1 if they are connected
                break
                
        # If we found nothing suitable, fallback to whole image (safety)
        if cv2.countNonZero(final_mask) == 0:
             return image_rgb, np.ones_like(gray) * 255
             
        # Mask the original image
        masked_img = cv2.bitwise_and(image_rgb, image_rgb, mask=final_mask)
        
        return masked_img, final_mask

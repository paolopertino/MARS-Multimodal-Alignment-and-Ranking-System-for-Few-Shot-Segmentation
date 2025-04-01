import abc

import numpy as np
import cv2

class VisualPromptGenerator(abc.ABC):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'draw') and
                callable(subclass.draw) or
                NotImplemented)
        
    def __init__(self) -> None:
        pass
    
    @abc.abstractmethod
    def draw(
        self, 
        image: np.ndarray,
        mask: np.ndarray, 
        color: tuple[int, int, int] = (255, 0, 0), 
        alpha: float = 0.5, 
        thickness: int = 2,
        zoom_percent: float = 0.
    ) -> np.ndarray:
        """Draw a visual prompt on an image.

        :param image: image to draw the prompt on
        :type image: np.ndarray
        :param mask: binary mask representing parts of the image to highlight with a visual prompt
        :type mask: np.ndarray
        :param color: color of the visual prompt, defaults to (255, 0, 0)
        :type color: tuple[int, int, int], optional
        :param alpha: transparency of the visual prompt on the original image. It ranges between 0 and 1, defaults to 0.5
        :type alpha: float, optional
        :param thickness: thickness of the visual prompt to apply to the image, defaults to 2
        :type thickness: int, optional
        :return: image with the visual prompt applied onto it
        :rtype: np.ndarray
        """
        raise NotImplementedError
    
    def zoom_on_masked_object(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        zoom_percent: float = 0.
    ) -> np.ndarray:
        """Zoom on the object(s) in an image based on a binary mask.

        :param image: input image to zoom on (RGB format) (pixel values in [0, 255])
        :type image: np.ndarray
        :param mask: binary mask of the object(s) to zoom on
        :type mask: np.ndarray
        :param zoom_percent: percentage of zoom to apply, defaults to 0.
        :type zoom_percent: float, optional
        :return: zoomed image
        :rtype: np.ndarray
        """
        if zoom_percent <= 0:
            return image
        
        # Ensure mask is binary
        mask = (mask > 0).astype(np.uint8)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image  # Return original image if no object is found

        # Find the bounding box that encompasses all contours
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))

        # Calculate the center of the bounding box
        center_x, center_y = x + w // 2, y + h // 2

        # Calculate the new dimensions based on the zoom percentage
        new_w = int(w * (100 / zoom_percent))
        new_h = int(h * (100 / zoom_percent))

        # Ensure the new dimensions don't exceed the image size
        new_w = min(new_w, image.shape[1])
        new_h = min(new_h, image.shape[0])

        # Calculate the new top-left corner
        new_x = max(0, center_x - new_w // 2)
        new_y = max(0, center_y - new_h // 2)

        # Adjust if the new bounding box exceeds image boundaries
        if new_x + new_w > image.shape[1]:
            new_x = image.shape[1] - new_w
        if new_y + new_h > image.shape[0]:
            new_y = image.shape[0] - new_h

        # Crop the region of interest
        cropped = image[new_y:new_y+new_h, new_x:new_x+new_w]

        # Resize the cropped image to match the original image dimensions
        result = cv2.resize(cropped, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        return result
    
class MaskGenerator(VisualPromptGenerator):
    def draw(
        self, 
        image: np.ndarray,
        mask: np.ndarray, 
        color: tuple[int, int, int] = (255, 0, 0), 
        alpha: float = 0.5, 
        thickness: int = 2,
        zoom_percent: float = 0.
    ) -> np.ndarray:
        """
        Blend an image X with a binary mask and a specified color according to the rule:
        composite_X = alpha * (mask * color) + (1 - alpha) * X
        
        :param image: Original image (3D numpy array for RGB images) (pixel values in [0, 255])
        :type image: np.ndarray
        :param mask: Binary mask (2D numpy array with values 0 or 1)
        :type mask: np.ndarray
        :param color: RGB color tuple for the mask (e.g., (255, 0, 0) for red), defaults to (255, 0, 0)
        :type color: tuple[int, int, int], optional
        :param alpha: Blending parameter between 0 and 1, defaults to 0.5
        :type alpha: float, optional
        :return: Composite image
        """
        # Ensure mask is binary
        mask = (mask > 0).astype(float)
        
        # Expand mask dimensions to match the image
        mask = np.expand_dims(mask, axis=-1)
        
        # Create a color mask
        color_mask = mask * np.array(color)
        
        # Create the composite image
        composite_image = alpha * color_mask + (1 - alpha) * image
        
        # Apply the blending only where the mask is non-zero
        result = np.where(mask, composite_image, image)
        result = result.astype(np.uint8)
        result = self.zoom_on_masked_object(result, mask, zoom_percent)
        
        return result
    
class BoundingBoxGenerator(VisualPromptGenerator):
    def draw(
        self, 
        image: np.ndarray,
        mask: np.ndarray, 
        color: tuple[int, int, int] = (255, 0, 0), 
        alpha: float = 0.5, 
        thickness: int = 2,
        zoom_percent: float = 0.
    ) -> np.ndarray:
        """Draw bounding boxes around objects in an image based on a binary mask.

        :param image: input image to draw bounding boxes on (RGB format) (pixel values in [0, 255])
        :type image: np.ndarray
        :param mask: binary mask of the objects to draw bounding boxes around
        :type mask: np.ndarray
        :param color: color of the bounding boxes drawn onto the input image, defaults to (255, 0, 0)
        :type color: tuple[int, int, int], optional
        :param alpha: Blending parameter between 0 and 1, defaults to 0.5
        :type alpha: float, optional
        :param thickness: thickness of the bounding box in pixel, defaults to 2
        :type thickness: int, optional
        :return: _description_
        :rtype: np.ndarray
        """
        # Ensure mask is binary and of type uint8
        mask = (mask > 0).astype(np.uint8)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a transparent overlay
        overlay = image.copy()

        # Draw filled bounding boxes on the overlay
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)

        # Blend the overlay with the original image
        result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        result = self.zoom_on_masked_object(result, mask, zoom_percent)

        return result
    
class MaskContourGenerator(VisualPromptGenerator):
    def draw(
        self, 
        image: np.ndarray,
        mask: np.ndarray, 
        color: tuple[int, int, int] = (255, 0, 0), 
        alpha: float = 0.5, 
        thickness: int = 2,
        zoom_percent: float = 0.
    ) -> np.ndarray:
        # Ensure mask is binary and of type uint8
        mask = (mask > 0).astype(np.uint8)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a transparent overlay
        overlay = image.copy()

        # Draw the contours on the overlay
        cv2.drawContours(overlay, contours, -1, color, thickness)

        # Blend the overlay with the original image
        result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        result = self.zoom_on_masked_object(result, mask, zoom_percent)

        return result
    
    
class EllipseGenerator(VisualPromptGenerator):
    def draw(
        self, 
        image: np.ndarray,
        mask: np.ndarray, 
        color: tuple[int, int, int] = (255, 0, 0), 
        alpha: float = 0.5, 
        thickness: int = 2,
        zoom_percent: float = 0.
    ) -> np.ndarray:
        expansion_factor = 1.2
        
        # Ensure mask is binary and of type uint8
        mask = (mask > 0).astype(np.uint8)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a transparent overlay
        overlay = image.copy()

        # Draw the ellipses on the overlay
        for contour in contours:
            # Calculate the rotated bounding box for the contour
            rect = cv2.minAreaRect(contour)
            center, axes, angle = rect

            # Calculate the expanded axes of the ellipse
            expanded_axes = (axes[0] * expansion_factor, axes[1] * expansion_factor)

            # Draw the ellipse on the overlay
            cv2.ellipse(overlay, (int(center[0]), int(center[1])), (int(expanded_axes[0] // 2), int(expanded_axes[1] // 2)), angle, 0, 360, color, thickness)

        # Blend the overlay with the original image
        result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        result = self.zoom_on_masked_object(result, mask, zoom_percent)

        return result
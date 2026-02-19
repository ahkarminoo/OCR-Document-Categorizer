import cv2
import numpy as np

def order_points(pts):
    """Orders coordinates: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def apply_perspective_transform(image, pts):
    """Warp the image to flatten the document."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate max width and height of the new flattened image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Destination points for the "birds eye view"
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Calculate the transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def clean_document(image_bytes):
    """Main pipeline: detects the document and crops it."""
    # 1. Convert uploaded bytes to an OpenCV image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    orig = image.copy()
    
    # 2. Grayscale, Blur, and Edge Detection to find the paper outline
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    
    # 3. Find the largest contours (shapes) in the image
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    screen_contour = None
    for c in cnts:
        # Approximate the polygon shape
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # If the shape has 4 points, we assume it's our document!
        if len(approx) == 4:
            screen_contour = approx
            break
            
    # 4. Apply the perspective warp if a document was found
    if screen_contour is not None:
        warped = apply_perspective_transform(orig, screen_contour.reshape(4, 2))
    else:
        warped = orig # Fallback: return original if no paper is detected
        
    # 5. Convert the cleaned image back to bytes so the API can return it
    _, buffer = cv2.imencode('.jpg', warped)
    return buffer.tobytes()
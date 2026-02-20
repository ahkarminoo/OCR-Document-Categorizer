import cv2
import numpy as np


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def apply_perspective_transform(image, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect
    maxWidth  = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)), 1)
    maxHeight = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)), 1)
    dst = np.array([[0, 0], [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def _resize_for_detection(image, max_side=1200):
    h, w  = image.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale == 1.0:
        return image, 1.0
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return resized, scale


def _candidate_score(contour, gray, img_area):
    area = cv2.contourArea(contour)
    if area <= 0:
        return -1
    rect      = cv2.minAreaRect(contour)
    box       = cv2.boxPoints(rect).astype(np.float32)
    rect_area = cv2.contourArea(box)
    if rect_area <= 0:
        return -1
    fill_ratio = area / rect_area
    area_ratio = area / img_area
    if area_ratio < 0.10 or fill_ratio < 0.45:
        return -1

    h, w = gray.shape[:2]
    x, y, bw, bh = cv2.boundingRect(contour.astype(np.int32))
    touches_all = (x <= int(0.02 * w) and y <= int(0.02 * h)
                   and (x + bw) >= int(0.98 * w) and (y + bh) >= int(0.98 * h))

    mask         = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour.astype(np.int32)], -1, 255, -1)
    inside_mean  = cv2.mean(gray, mask=mask)[0]
    outside_mean = cv2.mean(gray, mask=cv2.bitwise_not(mask))[0]
    brightness   = max(0.0, (inside_mean - outside_mean) / 255.0)

    score = area_ratio * 0.6 + fill_ratio * 0.25 + brightness * 0.15
    if touches_all:
        score -= 0.25
    return score


def _quad_covers_full_image(corners, image_shape, threshold=0.92):
    h, w = image_shape[:2]
    area = float(cv2.contourArea(corners.reshape(-1, 1, 2).astype(np.float32)))
    return area / (w * h) >= threshold


def _find_paper_corners_by_color(image):
    """Detect a white/cream paper region using HSV — best for docs on dark surfaces."""
    resized, scale = _resize_for_detection(image, max_side=1200)
    hsv            = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    h, w           = resized.shape[:2]
    img_area       = float(h * w)

    mask = cv2.inRange(hsv, (0, 0, 140), (180, 70, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5,  5),  np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    best_box, best_score = None, -1.0
    cx_img, cy_img = w * 0.5, h * 0.5

    for c in cnts:
        area       = cv2.contourArea(c)
        area_ratio = area / img_area
        if area_ratio < 0.08 or area_ratio > 0.95:
            continue
        rect       = cv2.minAreaRect(c)
        fill_ratio = area / max(1.0, rect[1][0] * rect[1][1])
        if fill_ratio < 0.40:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        dist  = np.sqrt((bx + bw * 0.5 - cx_img) ** 2 + (by + bh * 0.5 - cy_img) ** 2)
        max_d = np.sqrt(cx_img ** 2 + cy_img ** 2)
        score = area_ratio * 0.65 + fill_ratio * 0.20 + (1.0 - min(1.0, dist / max_d)) * 0.15
        if score > best_score:
            best_score = score
            best_box   = (c, rect)

    if best_box is None or best_score < 0.12:
        return None

    c, rect = best_box
    peri    = cv2.arcLength(c, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32") / scale
    return cv2.boxPoints(rect).astype(np.float32) / scale


def _collect_candidate_boxes(gray):
    candidates = []
    kernels    = [np.ones((7, 7), np.uint8), np.ones((11, 11), np.uint8)]

    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates.extend(cnts)

    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for k in kernels:
        cnts, _ = cv2.findContours(cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates.extend(cnts)

    adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 41, 10)
    for k in kernels:
        cnts, _ = cv2.findContours(cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, k),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates.extend(cnts)
    return candidates


def _find_document_corners(image):
    """Find document corners — HSV colour first, edge/threshold as fallback."""
    color_corners = _find_paper_corners_by_color(image)
    if color_corners is not None and not _quad_covers_full_image(color_corners, image.shape):
        return color_corners

    resized, scale = _resize_for_detection(image)
    gray           = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray           = cv2.GaussianBlur(gray, (3, 3), 0)
    img_area       = float(gray.shape[0] * gray.shape[1])
    best_score, best_box = -1.0, None

    for contour in _collect_candidate_boxes(gray):
        if len(contour) < 4:
            continue
        peri    = cv2.arcLength(contour, True)
        approx4 = None
        for eps in (0.01, 0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(contour, eps * peri, True)
            if len(approx) == 4:
                approx4 = approx
                break
        if approx4 is not None:
            score = _candidate_score(approx4.reshape(-1, 1, 2), gray, img_area)
            if score > best_score:
                best_score = score
                best_box   = approx4.reshape(4, 2).astype("float32")
        rect  = cv2.minAreaRect(contour)
        box   = cv2.boxPoints(rect).astype(np.float32)
        score = _candidate_score(box.reshape(-1, 1, 2), gray, img_area)
        if score > best_score:
            best_score = score
            best_box   = box

    if best_box is None or best_score < 0.18:
        return None
    if scale != 1.0:
        best_box = best_box / scale
    return best_box.astype("float32")


def _trim_warp_padding(image, pad_frac=0.005):
    """Remove thin black bars from perspective warp without cutting real content."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
    coords  = cv2.findNonZero(mask)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    ih, iw = image.shape[:2]
    sx, sy = max(0, int(pad_frac * iw)), max(0, int(pad_frac * ih))
    x0, y0 = max(x, sx), max(y, sy)
    x1, y1 = min(x + w, iw - sx), min(y + h, ih - sy)
    if (x1 - x0) < 100 or (y1 - y0) < 100:
        return image
    return image[y0:y1, x0:x1]


def preprocess_for_ocr(image):
    """Convert a cropped document to a high-contrast grayscale image for OCR."""
    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )


def clean_document(image_bytes: bytes) -> dict:
    """Detect and crop a document, then return OCR-ready images as bytes."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    orig  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if orig is None:
        raise ValueError("Unable to decode input image bytes.")

    corners           = _find_document_corners(orig)
    document_detected = corners is not None

    if document_detected and _quad_covers_full_image(corners, orig.shape):
        warped = orig
    elif document_detected:
        warped = apply_perspective_transform(orig, corners)
        warped = _trim_warp_padding(warped)
    else:
        warped = orig

    ocr_ready = preprocess_for_ocr(warped)
    _, cropped_buf = cv2.imencode(".jpg", warped)
    _, ocr_buf     = cv2.imencode(".png", ocr_ready)

    return {
        "cropped_image_bytes":  cropped_buf.tobytes(),
        "ocr_ready_image_bytes": ocr_buf.tobytes(),
        "document_detected":    document_detected,
    }

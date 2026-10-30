import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

image_folder = "date/imgs"
mask_folder = "date/masks"

def evaluate(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    gt_dilated = cv2.dilate(gt, kernel, iterations=1)

    TP = np.sum((pred == 1) & (gt_dilated == 1))
    FP = np.sum((pred == 1) & (gt_dilated == 0))
    FN = np.sum((pred == 0) & (gt == 1))
    TN = np.sum((pred == 0) & (gt_dilated == 0))

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    return precision, recall, f1, accuracy

results = []

if not os.path.exists(image_folder):
    print(f"Error: Folder {image_folder} not found.")
    exit()

for filename in os.listdir(image_folder):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    image_path = os.path.join(image_folder, filename)
    mask_name = filename.replace(".png", "_mask.png") # Adjusted based on your logic
    mask_path = os.path.join(mask_folder, mask_name)

    if not os.path.exists(mask_path):
        continue  

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    gt_edges = cv2.Canny(mask_bin, 100, 200)

    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_mag = np.uint8(np.clip(sobel_mag, 0, 255))

    _, sobel_bin = cv2.threshold(sobel_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    canny = cv2.Canny(blurred, 30, 100)

    sobel_metrics = evaluate(sobel_bin, gt_edges)
    canny_metrics = evaluate(canny, gt_edges)

    results.append((sobel_metrics, canny_metrics))

    if len(results) <= 3:
        plt.figure(figsize=(12, 6))
        imgs = [gray, mask_bin, gt_edges, sobel_bin, canny]
        lbls = ["Original", "Mask", "GT Edges", "Sobel (Otsu)", "Canny"]
        for i in range(5):
            plt.subplot(1, 5, i+1)
            plt.imshow(imgs[i], cmap='gray')
            plt.title(lbls[i])
            plt.axis('off')
        plt.show()

sobel_avg = np.mean([r[0] for r in results], axis=0)
canny_avg = np.mean([r[1] for r in results], axis=0)

print("\n" + "="*50)
print("FINAL CASE STUDY RESULTS (Average)")
print("="*50)
print(f"SOBEL:  Prec: {sobel_avg[0]:.4f} | Rec: {sobel_avg[1]:.4f} | F1: {sobel_avg[2]:.4f} | Acc: {sobel_avg[3]:.4f}")
print(f"CANNY:  Prec: {canny_avg[0]:.4f} | Rec: {canny_avg[1]:.4f} | F1: {canny_avg[2]:.4f} | Acc: {canny_avg[3]:.4f}")
print("="*50)
from PIL import Image
import numpy as np
import os
import re
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def load_column_from_csv(csv_path, column_name):
    df = pd.read_csv(csv_path)
    return df[column_name]

def find_indices_with_value(series, value):
    return series[series == value].index.tolist()

def calculate_confusion_matrix(csv_path, column_name, sorted_results):
    column_data = load_column_from_csv(csv_path, column_name)
 
    total_data_length = len(column_data)
    anomaly_indices = sorted_results
    scores = [0] * total_data_length
    for index in anomaly_indices:
        if 0 <= index < total_data_length:
            scores[index] = 1  
    auroc = roc_auc_score(column_data, scores)


    TP = FP = TN = FN = 0
    for i in range(total_data_length):
        if scores[i] == 1:
            if column_data[i] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if column_data[i] == 0:
                TN += 1
            else:
                FN += 1
                
    cm = [[TP, FN], [FP, TN]]
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = FP/(TN+FP)

    print("\nConfusion Matrix Table:")
    print(pd.DataFrame(cm, columns=["Predicted Positive", "Predicted Negative"], 
                       index=["Actual Positive", "Actual Negative"]))
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"FPR: {fpr:.4f}")
    print(f"AUROC: {auroc:.4f}")
    
    fpr, tpr, _ = roc_curve(column_data, scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auroc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig("ROC.png")


def find_largest_square(diagonal_pixels, img_array):
    max_area = 0
    largest_square = None

    num_exceeding = len(diagonal_pixels)
    for i in range(num_exceeding):
        for j in range(i + 1, num_exceeding):
            x1, y1 = diagonal_pixels[i]
            x2, y2 = diagonal_pixels[j]
            
            if x1 != x2 and y1 != y2:
                p3 = (x1, y2)
                p4 = (x2, y1)

                if 0 <= p3[0] < img_array.shape[1] and 0 <= p3[1] < img_array.shape[0] and \
                   0 <= p4[0] < img_array.shape[1] and 0 <= p4[1] < img_array.shape[0]:
                    
                    side_length = abs(x2 - x1)
                    area = side_length ** 2

                    if area > max_area:
                        max_area = area
                        largest_square = (x1, y1, x2, y2, p3, p4)

    return largest_square, max_area



def process_images_with_square_sum(image_folder, threshold):
    detected_indices = []

    for filename in os.listdir(image_folder):
        if filename.endswith('_st.tiff') and re.search(r'\d+_st\.tiff$', filename):
            image_path = os.path.join(image_folder, filename)
            image_index = re.search(r'(\d+)_st\.tiff$', filename).group(1)
            
            try:
                with Image.open(image_path) as img:
                    if img.mode not in ['L', 'F']:
                        raise ValueError("이미지 모드가 'L' 또는 'F'이어야 합니다.")
                    
                    img_array = np.array(img)
                    height, width = img_array.shape
                    diagonal_length = min(height, width)
                    exceeding_pixels = [(i, i) for i in range(diagonal_length) if img_array[i, i] >= threshold]
                    
                    if len(exceeding_pixels) < 2:
                        continue
                    
                    largest_square, _ = find_largest_square(exceeding_pixels, img_array)
                    
                    if largest_square:
                        x1, y1, x2, y2, _, _ = largest_square
                        square_sum = np.sum(img_array[min(y1, y2):max(y1, y2)+1, min(x1, x2):max(x1, x2)+1])
                        if square_sum >= threshold * ((abs(x2 - x1) + 1) ** 2):
                            detected_indices.append(int(image_index))
                            print(f"이미지 {image_index}: 사각형 내부의 총 픽셀값 합은 {square_sum:.2f}입니다.")
            except Exception as e:
                print(f"이미지 {image_index} 처리 중 오류 발생: {e}")

    return detected_indices

def process_images_with_all_squares(image_folder, threshold):
    detected_indices = []

    for filename in os.listdir(image_folder):
        if filename.endswith('_st.tiff') and re.search(r'\d+_st\.tiff$', filename):
            image_path = os.path.join(image_folder, filename)
            image_index = re.search(r'(\d+)_st\.tiff$', filename).group(1)
            
            try:
                with Image.open(image_path) as img:
                    if img.mode not in ['L', 'F']:
                        raise ValueError("이미지 모드가 'L' 또는 'F'이어야 합니다.")
                    
                    img_array = np.array(img)
                    height, width = img_array.shape
                    diagonal_length = min(height, width)
                    exceeding_pixels = [(i, i) for i in range(diagonal_length) if img_array[i, i] >= threshold]
                    
                    if len(exceeding_pixels) < 2:
                        continue
                    
                    found_anomaly = False
                    
                    for i in range(len(exceeding_pixels)):
                        for j in range(i + 1, len(exceeding_pixels)):
                            x1, y1 = exceeding_pixels[i]
                            x2, y2 = exceeding_pixels[j]
                            
                            if x1 != x2 and y1 != y2:
                                p3 = (x1, y2)
                                p4 = (x2, y1)

                                if 0 <= p3[0] < width and 0 <= p3[1] < height and \
                                   0 <= p4[0] < width and 0 <= p4[1] < height:

                                    square_sum = np.sum(img_array[min(y1, y2):max(y1, y2)+1, min(x1, x2):max(x1, x2)+1])
                                    
                                    if square_sum >= threshold * ((abs(x2 - x1) + 1) ** 2):
                                        detected_indices.append(int(image_index))
                                        print(f"이미지 {image_index}: 사각형 내부의 총 픽셀값 합은 {square_sum:.2f}입니다.")
                                        
                                        diagonal_pixels = []
                                        for k in range(min(abs(x2 - x1), abs(y2 - y1)) + 1):
                                            diag_pixel_value = img_array[min(y1, y2) + k, min(x1, x2) + k]
                                            diagonal_pixels.append(diag_pixel_value)
                                        
                                        print(f"이미지 {image_index}: 사각형 내부 주대각선의 픽셀값은 {diagonal_pixels}입니다.")
                                        found_anomaly = True
                                        break
                        
                        if found_anomaly:
                            break
                            
            except Exception as e:
                print(f"이미지 {image_index} 처리 중 오류 발생: {e}")

    return detected_indices

#st
def process_images(image_folder, threshold):
    detected_indices = []
    scale_factor = 232 / 1000

    for filename in os.listdir(image_folder):
        if filename.endswith('_st.tiff') and re.search(r'\d+_st\.tiff$', filename):
            image_path = os.path.join(image_folder, filename)
            image_index = re.search(r'(\d+)_st\.tiff$', filename).group(1)
            
            try:
                with Image.open(image_path) as img:
                    if img.mode not in ['L', 'F']:
                        raise ValueError("이미지 모드가 'L' 또는 'F'이어야 합니다.")
                    
                    img_array = np.array(img)
                    height, width = img_array.shape
                    diagonal_length = min(height, width)
                    exceeding_pixels = [(i, i) for i in range(diagonal_length) if img_array[i, i] >= threshold]
                    
                    if len(exceeding_pixels) < 2:
                        continue
                    
                    largest_square, _ = find_largest_square(exceeding_pixels, img_array)
                    
                    if largest_square:
                        detected_indices.append(int(image_index))
            except Exception as e:
                print(f"이미지 {image_index} 처리 중 오류 발생: {e}")

    return detected_indices

def get_image_sizes(base_folder):
    img_sizes = []

    for root, _, files in os.walk(base_folder):
        for filename in sorted(files, key=lambda x: int(re.search(r'(\d+)\.tiff$', x).group(1)) if re.search(r'(\d+)\.tiff$', x) else float('inf')):
            if filename.endswith('.tiff') and not filename.endswith('_st.tiff') and not filename.endswith('_ae.tiff'):
                image_path = os.path.join(root, filename)
                
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size

                        img_sizes.append(min(width, height))
                        
                        
                except Exception as e:
                    print(f"이미지 {filename} 처리 중 오류 발생: {e}")

    return img_sizes

def process_images_with_line_sum(base_folder, threshold, img_sizes):
    detected_indices = []
    result_values = []
    for root, _, files in os.walk(base_folder):
        for filename in files:
            if filename.endswith('.tiff') and not filename.endswith('_st.tiff') and not filename.endswith('_ae.tiff'):
                image_path = os.path.join(root, filename)
                match = re.search(r'(\d+)\.tiff$', filename)
                if match:
                    image_index = int(match.group(1))
                else:
                    print(f"잘못된 파일 이름 형식: {filename}")
                    continue

                try:
                    with Image.open(image_path) as img:
                        if img.mode not in ['L', 'F']:
                            raise ValueError("이미지 모드가 'L' 또는 'F'이어야 합니다.")
                        
                        img_array = np.array(img)
                        
                        height, width = img_array.shape
                        diagonal_length = min(height, width)
                        exceeding_pixels = [(i, i) for i in range(diagonal_length) if img_array[i, i] >= threshold]
                        
                        max_total_sum = 0
                        max_x = None
                        anomaly_x_coords = []

                        for (x, y) in exceeding_pixels:
                            row_sum = np.sum(img_array[y, :])
                            col_sum = np.sum(img_array[:, x])
                            
                            total_sum = row_sum + col_sum - img_array[y, x]
                            total_pixels = height + width - 1

                            if total_sum > max_total_sum:
                                max_total_sum = total_sum
                                max_x = x

                            if total_sum >= total_pixels * threshold:
                                anomaly_x_coords.append(x)
                                detected_indices.append((image_index, x))
                                
                                cumulative_sum_before_index = sum(img_sizes[:image_index-1])
                                result_value = cumulative_sum_before_index + x
                                result_values.append(result_value)
                        
                except Exception as e:
                    print(f"이미지 {image_index} 처리 중 오류 발생: {e}")

    return detected_indices, result_values

def process_images_with_line_sum_max(base_folder, threshold, img_sizes, max_img_sizes):
    detected_indices = []
    result_values = []

    for root, _, files in os.walk(base_folder):
        for filename in files:
            if filename.endswith('.tiff') and not filename.endswith('_st.tiff') and not filename.endswith('_ae.tiff'):
                image_path = os.path.join(root, filename)
                match = re.search(r'(\d+)\.tiff$', filename)
                if match:
                    image_index = int(match.group(1))
                else:
                    print(f"잘못된 파일 이름 형식: {filename}")
                    continue

                try:
                    with Image.open(image_path) as img:
                        if img.mode not in ['L', 'F']:
                            raise ValueError("이미지 모드가 'L' 또는 'F'이어야 합니다.")
                        
                        img_array = np.array(img)
                        height, width = img_array.shape

                        if height >= max_img_sizes and width >= max_img_sizes:
                            for x in range(width):
                                detected_indices.append((image_index, x))
                                cumulative_sum_before_index = sum(img_sizes[:image_index - 1])
                                result_value = cumulative_sum_before_index + x
                                result_values.append(result_value)
                            continue  

                        diagonal_length = min(height, width)
                        exceeding_pixels = [(i, i) for i in range(diagonal_length) if img_array[i, i] >= threshold]
                        
                        max_total_sum = 0
                        average_sum = 0
                        max_x = None
                        anomaly_x_coords = []

                        for (x, y) in exceeding_pixels:
                            row_sum = np.sum(img_array[y, :])
                            col_sum = np.sum(img_array[:, x])

                            total_sum = row_sum + col_sum - img_array[y, x]
                            total_pixels = height + width - 1


                            if total_sum >= total_pixels * threshold:
                                anomaly_x_coords.append(x)
                                detected_indices.append((image_index, x))

                                cumulative_sum_before_index = sum(img_sizes[:image_index-1])
                                result_value = cumulative_sum_before_index + x
                                result_values.append(result_value)
                                
                        
                except Exception as e:
                    print(f"이미지 {image_index} 처리 중 오류 발생: {e}")

    return detected_indices, result_values


def process_images_with_line_sum_max_graph(base_folder, threshold, img_sizes, max_img_sizes):
    detected_indices = []
    result_values = [] 
    anomaly_values = [] 

    for root, _, files in os.walk(base_folder):
        for filename in files:
            if filename.endswith('.tiff') and not filename.endswith('_st.tiff') and not filename.endswith('_ae.tiff'):
                image_path = os.path.join(root, filename)
                match = re.search(r'(\d+)\.tiff$', filename)
                if match:
                    image_index = int(match.group(1))
                else:
                    print(f"잘못된 파일 이름 형식: {filename}")
                    continue

                try:
                    with Image.open(image_path) as img:
                        if img.mode not in ['L', 'F']:
                            raise ValueError("이미지 모드가 'L' 또는 'F'이어야 합니다.")
                        
                        img_array = np.array(img)
                        height, width = img_array.shape

                        if height >= max_img_sizes and width >= max_img_sizes:
                            for x in range(width):
                                detected_indices.append((image_index, x))
                                cumulative_sum_before_index = sum(img_sizes[:image_index - 1])
                                result_value = cumulative_sum_before_index + x
                                result_values.append(result_value)
                            continue  

                        diagonal_length = min(height, width)
                        exceeding_pixels = [(i, i) for i in range(diagonal_length) if img_array[i, i] >= threshold]
                        
                        max_total_sum = 0
                        average_sum = 0
                        max_x = None
                        anomaly_x_coords = []

                        for (x, y) in exceeding_pixels:
                            row_sum = np.sum(img_array[y, :])
                            col_sum = np.sum(img_array[:, x])
                            
                            val = img_array[y, x]
                            total_sum = row_sum + col_sum - img_array[y, x]
                            total_pixels = height + width - 1

                            if total_sum > max_total_sum:
                                max_total_sum = total_sum
                                max_x = x
                                
                            if total_sum >= total_pixels * threshold:
                                anomaly_x_coords.append(x)
                                detected_indices.append((image_index, x))

                                cumulative_sum_before_index = sum(img_sizes[:image_index-1])
                                result_value = cumulative_sum_before_index + x
                                result_values.append(result_value)

                except Exception as e:
                    print(f"이미지 {image_index} 처리 중 오류 발생: {e}")

    plt.figure(figsize=(10, 6))
    plt.hist(anomaly_values, bins=20, alpha=0.7, color='red', label='Anomaly Values')
    plt.title('Distribution of Anomaly Values')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig("Anomaly_Values_Distribution.png")
    plt.show()

    return detected_indices, result_values


def calculate_metrics(actual_indices, detected_indices, total_images):
    actual = [1 if i in actual_indices else 0 for i in range(1, total_images + 1)]
    predicted = [1 if i in detected_indices else 0 for i in range(1, total_images + 1)]
    
    cm = confusion_matrix(actual, predicted)
    
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted, zero_division=0)
    recall = recall_score(actual, predicted, zero_division=0)
    f1 = f1_score(actual, predicted, zero_division=0)
    
    return cm, accuracy, precision, recall, f1

def process_and_merge_folders(base_folder, threshold, actual_indices, total_images):
    all_detected_indices = []
    
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            print(f"\nProcessing folder: {folder_name}")
            
            detected_indices, results = process_images_with_line_sum(folder_path, threshold)
            all_detected_indices.extend(detected_indices)
    
    print(results)
    all_detected_indices = list(set(all_detected_indices))
    
    cm, accuracy, precision, recall, f1 = calculate_metrics(actual_indices, all_detected_indices, total_images)
    
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    
def calculate_confusion_matrix_graph(csv_path, column_name, sorted_results, average_sums):
    column_data = load_column_from_csv(csv_path, column_name)
    
    total_data_length = len(column_data)
    anomaly_indices = sorted_results
    scores = [0] * total_data_length
    for index in anomaly_indices:
        if 0 <= index < total_data_length:
            scores[index] = 1
            
    auroc = roc_auc_score(column_data, scores)

    TP = FP = TN = FN = 0
    actual_positive_indices = []
    
    for i in range(total_data_length):
        if scores[i] == 1:
            if column_data[i] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if column_data[i] == 0:
                TN += 1
            else:
                FN += 1

    for i in range(total_data_length):
        if column_data[i] == 1:
            actual_positive_indices.append(i)
            
    cm = [[TP, FN], [FP, TN]]
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = FP / (TN + FP)

    print("\nConfusion Matrix Table:")
    print(pd.DataFrame(cm, columns=["Predicted Positive", "Predicted Negative"], 
                       index=["Actual Positive", "Actual Negative"]))
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"FPR: {fpr:.4f}")
    print(f"AUROC: {auroc:.4f}")

    fpr, tpr, _ = roc_curve(column_data, scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auroc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig("ROC.png")

    actual_positive_sums = [average_sums[i] for i in actual_positive_indices]

    plt.figure(figsize=(10, 6))
    plt.hist(actual_positive_sums, bins=20, alpha=0.7, label='Actual Positive Average Sums', color='purple')
    plt.title('Distribution of Actual Positive Average Sums')
    plt.xlabel('Average Sum')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig("Actual_Positive_Average_Sums.png")
    plt.show()
    

csv_path = 'path'
base_folder = 'path'
column_name = 'column'

encoding = 'LGAF'
encoding2 = 'GGAF'

max_img_size = 2000

imgsize = get_image_sizes(base_folder)

threshold = 0.1

indices, results = process_images_with_line_sum_max(base_folder, threshold, imgsize, max_img_size)
print("calculating all_anomalies..")
calculate_confusion_matrix(csv_path, column_name, sorted(results))

print()

import argparse
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from segmentation_models_pytorch import DeepLabV3Plus
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import shutil
def parse_args():
    parser = argparse.ArgumentParser(description="Person and Crosswalk Detection Script")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to input image directory")
        
    return parser.parse_args()
    
def clear_directory(directory_path):
    print('##clear directory_path##', directory_path)
    # 디렉토리 내의 모든 파일 및 폴더 리스트를 가져옵니다.
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        # 파일이면 삭제합니다.
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
        # 디렉토리면 내부의 모든 내용을 삭제합니다.
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def train(img):
    data = 'data'
    lastmask='Last_Mask'
    clear_directory(data)
    clear_directory(lastmask)


    
    image_dir = img


    print(image_dir)
    mask_output_dir = 'Last_Mask'
    os.makedirs(mask_output_dir,exist_ok=True)

    transform = A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    seg_model = DeepLabV3Plus(encoder_name="resnet34", in_channels=3, classes=4)
    seg_model.load_state_dict(torch.load(f'model/deeplabv2_best_model.pth'))
    seg_model.cuda()
    seg_model.eval()

    def segment_and_save_images(image_dir, mask_output_dir):
        for filename in tqdm(sorted(os.listdir(image_dir))):
            image_path = os.path.join(image_dir, filename)
            image = np.array(Image.open(image_path).convert("RGB"))

            # 이미지 변환 및 모델 추론
            augmented = transform(image=image)
            image_tensor = augmented["image"].unsqueeze(0).cuda()
            with torch.no_grad():
                output = seg_model(image_tensor)
                preds = torch.argmax(torch.softmax(output, dim=1), dim=1).cpu().squeeze(0).numpy()

            # 결과 마스크를 원본 크기로 변환
            mask_resized = Image.fromarray(preds.astype(np.uint8)).resize(image.shape[1::-1], Image.NEAREST)
            mask_resized = np.array(mask_resized)

            # 결과 저장
            result_path = os.path.join(mask_output_dir, f"{os.path.splitext(filename)[0]}_mask.png")
            Image.fromarray(mask_resized.astype(np.uint8)).save(result_path)

    # 실행
    segment_and_save_images(image_dir, mask_output_dir) 




    # 이미지 및 마스크 경로 설정 (공통 경로로 통합)

    crosswalk_model = YOLO("model/best_yolov_cross_11m.pt")

    for img_name in os.listdir(f'{image_dir}'):
        imgpath = os.path.join(f'{image_dir}',img_name)
        maskname = img_name.replace('.jpg','_mask.png')
        
        maskpath = os.path.join(f'{mask_output_dir}',maskname)
        
        # 이미지 경로 및 예측 수행 (횡단보도 탐지용 모델 사용)
        crosswalk_results = crosswalk_model.predict(imgpath, imgsz=1800, conf=0.05)
        # 마스크 이미지 로드
        mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)

        # 클래스 1과 2에 해당하는 픽셀만 추출 (마스크 값이 1 또는 2인 부분만)
        class_1_2_mask = np.where((mask == 1) | (mask == 2), 255, 0).astype(np.uint8)

        # 원본 이미지 로드
        img = cv2.imread(imgpath)
        height, width = img.shape[:2]

        # 횡단보도 마스크 생성 (횡단보도 탐지 결과 바운딩 박스 영역을 이진 마스크로 생성)
        crosswalk_mask = np.zeros((height, width), dtype=np.uint8)
        for result in crosswalk_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # 바운딩 박스 좌표를 정수로 변환
                # 바운딩 박스에 해당하는 영역을 흰색(255)으로 채우기
                cv2.rectangle(crosswalk_mask, (x1, y1), (x2, y2), 255, -1)  # 흰색(255)으로 채움

        # 사람 탐지용 YOLO 모델 로드
        model = YOLO("model/best_yolov11n.pt")

        # 이미지 경로 및 예측 수행 (사람 탐지용 모델 사용)
        results = model.predict(imgpath, imgsz=1800, conf=0.05)

        # 예측 결과에서 사람 클래스(클래스 ID 0)에 해당하는 바운딩 박스만 필터링
        person_boxes = []
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # 클래스 ID 0은 사람을 의미함
                    person_boxes.append(box.xyxy[0].cpu().numpy())  # 바운딩 박스 좌표 저장

        # 도로 위에 있는 사람 바운딩 박스만 필터링 및 시각화, 그리고 발 좌표 저장
        output_data = []
        for box in person_boxes:
            x1, y1, x2, y2 = map(int, box)  # 바운딩 박스 좌표를 정수로 변환
            foot_x = int((x1 + x2) / 2)  # 발의 중간 좌표 (x 방향)
            foot_y = int(y2)  # 발의 y 좌표는 바운딩 박스의 아래쪽

            # 발의 좌표가 도로 마스크에서 흰색(255)인 경우 도로 위에 있는 것으로 간주
            if 0 <= foot_y < class_1_2_mask.shape[0] and 0 <= foot_x < class_1_2_mask.shape[1]:
                if class_1_2_mask[foot_y, foot_x] == 255:
                    # 도로 위에 있는 사람의 바운딩 박스를 횡단보도 영역에 대해 추가적으로 검사
                    if crosswalk_mask[foot_y, foot_x] == 255:
                        output_data.append(f"1 {x1} {y1} {x2} {y2}")
                    else:
                        output_data.append(f"0 {x1} {y1} {x2} {y2}")

        # 결과를 텍스트 파일로 저장
        img_name = img_name.replace('.jpg','.txt')
        os.makedirs('data',exist_ok=True)
        output_path = os.path.join('data',img_name)
        print(output_path)
        with open(output_path, "w") as f:
            for line in output_data:
                f.write(line + "\n")

if __name__ == "__main__":
    args = parse_args()
    print(f"이미지 디렉토리: {args.image_dir}")
    train(args.image_dir)
# def main():
#     args = parse_args()
#     print(f"이미지 디렉토리: {args.image_dir}")

#     train(args.image_dir)

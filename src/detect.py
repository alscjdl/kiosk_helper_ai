import cv2
import easyocr
import torch
from ultralytics import YOLO

# [1. 초기화 영역] 모델 및 OCR 리더기는 처음 실행할 때 딱 한 번만 메모리에 올림
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 현재 연산 디바이스: {device.upper()}")

model = YOLO("../models/best.pt")
reader = easyocr.Reader(["ko", "en"], gpu=(device == "cuda"))


# ==========================================
# EasyOCR 기반 텍스트 추출 함수
# ==========================================
def extract_text_from_button(cropped_image):
    """
    잘린 버튼 이미지를 입력받아 글자(텍스트)만 추출하여 반환하는 함수
    """
    if cropped_image is None or cropped_image.size == 0:
        return ""

    # EasyOCR로 글자 읽기
    ocr_result = reader.readtext(cropped_image)

    if ocr_result:
        # 공백을 제거한 텍스트 반환 (예: "결제 하기" -> "결제하기")
        btn_text = ocr_result[0][1].replace(" ", "")
        return btn_text
    
    return ""


# ==========================================
# 🔄 메인 파이프라인 실행 영역
# ==========================================
if __name__ == "__main__":
    image_path = "../images/test.jpg"
    
    # YOLO 추론
    results = model(image_path, conf=0.5, device=device, save=False)
    img = cv2.imread(image_path)

    print("\n=== 검출 및 글자 인식 결과 ===")
    detected = []
    payment_buttons_text = []

    for box in results[0].boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        label = results[0].names[cls]

        detected.append(label)
        print(f"\n[검출] {label} ({conf:.2f})")

        # payment_button일 때만 이미지 크롭 후 함수 호출!
        if label == "payment_button":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_btn = img[y1:y2, x1:x2]

            # ⭐️ 독립 함수를 매칭해서 사용
            btn_text = extract_text_from_button(cropped_btn)
            
            if btn_text:
                payment_buttons_text.append(btn_text)
                print(f"       ㄴ 🔎 버튼 속 실제 글자: '{btn_text}'")
            else:
                print("       ㄴ 🔎 버튼 속 글자를 읽지 못함")

    print("\n==============================")
    print("검출된 클래스 목록:", detected)
    print("검출된 결제 버튼 글자들:", payment_buttons_text)
    print("==============================")

    # ----------------------
    # 상황 판단 및 가이드 텍스트 정의
    # ----------------------
    guide_text = "Guide: Analyzing..."

    if "payment_button" in detected and "cart_area" in detected:
        print("\n현재 단계: 장바구니 확인 및 결제 단계")
        guide_text = "Guide: Check your cart and payment!"
        if "결제" in payment_buttons_text or "결제하기" in payment_buttons_text:
            guide_text = "Guide: Touch the [PAYMENT] button below!"
    elif "menu_area" in detected:
        print("\n현재 단계: 메뉴 선택 단계")
        guide_text = "Guide: Choose your menu from the screen."
    elif "back_button" in detected:
        print("\n현재 단계: 이전 화면 이동 가능")
        guide_text = "Guide: You can go back to the previous page."
    else:
        print("\n현재 단계: 알 수 없음")
        guide_text = "Guide: Follow the instructions on the screen."

    # ----------------------
    #시스템 통합 및 시각화 (인터페이스 반영용 파일 저장)
    # ----------------------
    h, w, c = img.shape
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls)
        label = results[0].names[cls]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.rectangle(img, (0, h - 80), (w, h), (0, 0, 0), -1)
    cv2.putText(img, guide_text, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imwrite("../images/result_guide.jpg", img)
    print(f"🎯 파이프라인 정상 작동 완료!")
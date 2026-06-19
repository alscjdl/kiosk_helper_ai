import cv2
import easyocr
import torch
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# 경로 설정
# ==========================================
# 실행 파일(__file__) 위치 기준으로 상위 폴더 구조를 자동 계산
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best.pt"
IMAGE_PATH = BASE_DIR / "images" / "test.jpg"
OUTPUT_PATH = BASE_DIR / "images" / "result_guide.jpg"

# ==========================================
# 모델 및 OCR 초기화
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 현재 연산 디바이스: {device.upper()}")

model = YOLO(str(MODEL_PATH))
reader = easyocr.Reader(["ko", "en"], gpu=(device == "cuda"))


# ==========================================
#  EasyOCR 기반 텍스트 추출 함수
# ==========================================
def extract_text_from_button(cropped_image):
    if cropped_image is None or cropped_image.size == 0:
        return ""

    try:
        ocr_result = reader.readtext(cropped_image)

        if not ocr_result:
            return ""

        # 검출된 모든 텍스트 조각을 모아서 공백 없이 하나로 합침
        texts = [item[1] for item in ocr_result]
        return "".join(texts).replace(" ", "")

    except Exception as e:
        print(f"❌ OCR 오류 발생: {e}")
        return ""


# ==========================================
# 상황 분석 및 가이드 생성 함수
# ==========================================
def generate_guide(detected, payment_buttons_text):
    guide_text = "Guide: Analyzing..."  # 기본 가이드 초기화

    if "payment_button" in detected and "cart_area" in detected:
        print("\n현재 단계: 장바구니 확인 및 결제 단계")
        guide_text = "Guide: Check your cart and payment!"
        
        # 전과 똑같이 "결제" 또는 "결제하기" 글자가 포함되어 있는지 체크
        if "결제" in payment_buttons_text or "결제하기" in payment_buttons_text:
            print("💡 가이드 안내: '결제하기' 버튼을 누르도록 강조하세요.")
            guide_text = "Guide: Touch the [PAYMENT] button below!"

    elif "menu_area" in detected:
        print("\n현재 단계: 메뉴 선택 단계")
        guide_text = "Guide: Choose your menu from the screen."

    elif "back_button" in detected:
        print("\n현재 단계: 이전 화면 이동 가능")
        guide_text = "Guide: You can go back to the previous page."

    else:
        print("\n현재 단계: 알 수 없음 (안내 및 카드 삽입 지시 화면 등)")
        guide_text = "Guide: Follow the instructions on the screen."

    return guide_text
# ==========================================
# 메인 실행 파이프라인
# ==========================================
if __name__ == "__main__":
    # 1. 원본 이미지 로드
    img = cv2.imread(str(IMAGE_PATH))

    if img is None:
        raise FileNotFoundError(f"❌ 이미지를 찾을 수 없습니다: {IMAGE_PATH}")

    # 2. YOLO 추론 (디스크 중복 저장 방지로 속도 최적화)
    results = model(str(IMAGE_PATH), conf=0.5, device=device, save=False)

    detected = set()
    payment_buttons_text = []

    print("\n=== 검출 및 OCR 결과 ===")

    # 3. 바운딩 박스 순회 및 크롭/OCR 처리
    for box in results[0].boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        label = results[0].names[cls]

        detected.add(label)
        print(f"\n[검출] {label} ({conf:.2f})")

        # 버튼일 경우 해당 영역만 잘라내어 OCR 수행
        if label == "payment_button":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_btn = img[y1:y2, x1:x2]

            btn_text = extract_text_from_button(cropped_btn)

            if btn_text:
                payment_buttons_text.append(btn_text)
                print(f"       ㄴ 🔎 OCR 결과: '{btn_text}'")

    print("\n==============================")
    print("검출된 클래스 목록:", detected)
    print("결제 버튼 텍스트 모음:", payment_buttons_text)
    print("==============================")

    # 4. 가이드 텍스트 생성
    guide_text = generate_guide(detected, payment_buttons_text)
    print("\n[생성된 가이드 자막] " + guide_text)

    # ==========================================
    #  시각화 및 최종 저장
    # ==========================================
    h, w, _ = img.shape

    # 이미지 위에 모든 객체 박스 그리기
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls)
        label = results[0].names[cls]

        # 초록색 박스 칠하기
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 하단에 검은색 안내 자막 바 그리기
    cv2.rectangle(img, (0, h - 80), (w, h), (0, 0, 0), -1)

    # 검은 바 위에 가이드 자막 흰색 글씨로 쓰기
    cv2.putText(img, guide_text, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 최종 가이드 결과 이미지 저장
    cv2.imwrite(str(OUTPUT_PATH), img)
    print(f"\n🎯 결과 이미지 저장 완료: {OUTPUT_PATH}")
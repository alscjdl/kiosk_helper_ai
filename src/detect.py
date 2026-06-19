import cv2
import easyocr
from ultralytics import YOLO

# 1. 모델 및 OCR 리더기 초기화
model = YOLO("../models/best.pt")
reader = easyocr.Reader(["ko", "en"])  # 한국어, 영어 글자 읽기 설정

# 2. 이미지 추론 및 이미지 원본 불러오기
image_path = "../images/test.jpg"
results = model(image_path, conf=0.5, save=True)
img = cv2.imread(image_path)  # 글자를 오려내기 위해 OpenCV로 이미지를 읽음

print("=== 검출 및 글자 인식 결과 ===")

detected = []
payment_buttons_text = []  # 결제 버튼들의 실제 글자를 모아둘 리스트

# 3. YOLO가 찾은 박스들을 하나씩 돌면서 처리
for box in results[0].boxes:
    cls = int(box.cls)
    conf = float(box.conf)
    label = results[0].names[cls]

    detected.append(label)
    print(f"\n[검출] {label} ({conf:.2f})")

    # [핵심 추가] 찾은 객체가 payment_button인 경우, 글자까지 읽기
    if label == "payment_button":
        # 버튼의 네모 좌표 구하기
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # 이미지에서 버튼 영역만 싹둑 자르기 (Crop)
        cropped_btn = img[y1:y2, x1:x2]

        # 자른 버튼 이미지에서 글자 추출하기
        ocr_result = reader.readtext(cropped_btn)

        if ocr_result:
            # 띄어쓰기 없애고 글자만 추출 (예: "멤버십 혜택" -> "멤버십혜택")
            btn_text = ocr_result[0][1].replace(" ", "")
            payment_buttons_text.append(btn_text)
            print(f"       ㄴ 🔎 버튼 속 실제 글자: '{btn_text}'")
        else:
            print("       ㄴ 🔎 버튼 속 글자를 읽지 못함")


print("\n==============================")
print("검출된 클래스 목록:", detected)
print("검출된 결제 버튼 글자들:", payment_buttons_text)
print("==============================")


# ----------------------
# 상황 판단 로직 (OCR 데이터 활용하도록 업그레이드 가능!)
# ----------------------

if "payment_button" in detected and "cart_area" in detected:
    print("\n현재 단계: 장바구니 확인 및 결제 단계")
    # 업그레이드 팁: "결제"라는 글자가 버튼 목록에 있으면 진짜 최종 결제창이라고 확신 가능!
    if "결제" in payment_buttons_text or "결제하기" in payment_buttons_text:
        print("💡 가이드 안내: '결제하기' 버튼을 누르도록 강조하세요.")

elif "menu_area" in detected:
    print("\n현재 단계: 메뉴 선택 단계")

elif "back_button" in detected:
    print("\n현재 단계: 이전 화면 이동 가능")

else:
    print("\n현재 단계: 알 수 없음 (안내 및 카드 삽입 지시 화면 등)")
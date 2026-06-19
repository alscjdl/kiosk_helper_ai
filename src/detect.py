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
# 상황 판단 로직
# ----------------------
guide_text = "Guide: Analyzing..."  # 시각화 화면에 띄울 안내 멘트 변수

if "payment_button" in detected and "cart_area" in detected:
    print("\n현재 단계: 장바구니 확인 및 결제 단계")
    guide_text = "Guide: Check your cart and payment!"
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


# ----------------------
# [여기서부터 새로 추가된 시각화 섹션!]
# ----------------------
print("\n[시각화 작업 시작] 이미지 위에 안내선과 가이드를 그립니다...")

# 1. YOLO가 검출한 모든 상자 화면에 그리기
for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls = int(box.cls)
    label = results[0].names[cls]
    
    # 초록색 사각형(BGR: 0, 255, 0) 두께 3으로 그리기
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    # 사각형 위에 라벨 이름 쓰기
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 2. 화면 맨 밑에 할머니가 보기 편하게 '검은색 안내 자막 바' 깔기
h, w, c = img.shape
cv2.rectangle(img, (0, h - 80), (w, h), (0, 0, 0), -1)  # -1은 내부를 꽉 채운다는 뜻

# 3. 상황 판단 결과에 맞게 갱신된 guide_text를 검은 바 위에 흰색 글씨로 쓰기
cv2.putText(img, guide_text, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# 4. 가이드가 완성된 이미지를 새 파일로 저장하기
output_path = "../images/result_guide.jpg"
cv2.imwrite(output_path, img)

print(f"🎯 성공! 시각화 완료 이미지가 '{output_path}'에 저장되었습니다.")
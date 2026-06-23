import cv2
import easyocr
import torch
import os
from pathlib import Path
from ultralytics import YOLO
from openai import OpenAI
from dotenv import load_dotenv

# ==========================================
# 경로 설정 및 환경 변수 로드 (.env)
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

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
#  OpenAI 클라이언트 안전한 초기화
# ==========================================
openai_key = os.getenv("OPENAI_API_KEY")

if openai_key:
    client = OpenAI(api_key=openai_key)
    print("🔑 OpenAI API Key 로드 완료.")
else:
    client = None
    print("⚠️ 경고: .env 파일에 OPENAI_API_KEY가 없습니다. GPT 기능이 제한됩니다.")


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

        texts = [item[1] for item in ocr_result]
        return "".join(texts).replace(" ", "")

    except Exception as e:
        print(f"❌ OCR 오류 발생: {e}")
        return ""


# ==========================================
#  GPT 기반 친절한 안내 멘트 생성 함수
# ==========================================
def call_friendly_gpt_guide(situation_context, actual_buttons_text):
    # API 키가 없거나 클라이언트 생성에 실패했을 경우 예외 처리
    if client is None:
        return "화면 분석 안내를 일시적으로 이용할 수 없습니다. 화면의 안내를 따라 천천히 진행해 주세요."

    try:
        buttons_str = ", ".join(actual_buttons_text) if actual_buttons_text else "없음"

        response = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 키오스크 조작에 어려움을 겪는 모든 디지털 정보 약자를 돕는 아주 다정하고 친절한 무인 안내원이야.\n\n"
                        "[⚠️ 필수 규칙 - 오타 교정 및 외래어 금지]\n"
                        "1. 카카오페이, 제로페이 등 [실제 화면 글자 목록]에 없는 결제 수단을 상상해서 지어내지 마.\n"
                        "2. ⭐[범용적 장소 선택 규칙]: 현재 화면 상황이 '식사 장소(매장/포장) 선택 단계'라면, "
                        "글자 목록에 어떤 오타나 깨진 글자(예: '다장', '배장', '포잠' 등)가 들어오더라도 낚이지 말고, "
                        "이 화면이 '매장에서 먹기'와 '포장하기' 중 하나를 고르는 첫 화면임을 문맥으로 눈치채서 자연스럽게 안내해 줘.\n"
                        "3. 글자 목록에 오타나 깨진 글자가 있다면 문맥을 파악하여 원래 어떤 단어였을지 눈치껏 올바르게 교정해서 안내해 줘. "
                        "단, 가이드 문장 안에는 오타를 그대로 노출하지 말고 올바른 우리말 단어만 사용해.\n"
                        "4. '카테고리', '탭', '터치', '클릭' 같은 어려운 외래어는 '손가락으로 살짝 누르기', '칸' 같은 쉬운 우리말로 순화해.\n"
                        "5. 특정 대상을 지칭하는 호칭(할머니, 어르신 등)은 절대 쓰지 말고 본론만 다정하게 작성해.\n"
                        "6. 정중하면서도 다정한 어조 (~세요, ~하시면 돼요)로 딱 2~3문장 이내로 깔끔하게 작성해."
                    )
                },
                {
                    "role": "user",
                    "content": f"현재 화면 상황: {situation_context}\n[실제 화면 글자 목록]: {buttons_str}\n\n이 조건에 맞춰 자연스럽고 정확한 가이드 문장을 만들어줘."
                }
            ],
            temperature=0.3  
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠️ GPT API 호출 실패: {e}")
        return "화면을 분석하고 있습니다. 잠시만 기다려 주시면 안내해 드릴게요."


# ==========================================
# [통합] 규칙 기반 상황 분석 및 GPT 가이드 연동 함수
# ==========================================
def generate_guide(detected, payment_buttons_text):
    situation_context = "현재 키오스크 화면을 분석하고 있는 단계입니다."

    # 🚨 [1순위] 장바구니 결제 단계 (우선순위 최고)
    if "payment_button" in detected and "cart_area" in detected:
        print("\n현재 단계: 장바구니 확인 및 결제 단계")
        situation_context = "고객이 주문할 메뉴들을 장바구니에 담은 후, 담긴 내역을 확인하고 결제를 진행해야 하는 단계"
        
        # ⚠️ [피드백 2 반영] 여러 개 텍스트 중 하나라도 "결제" 단어가 포함되어 있는지 검사
        if any("결제" in text for text in payment_buttons_text):
            situation_context = "화면 하단에 '결제' 또는 '결제하기'라고 적힌 큰 버튼이 명확하게 보이는 장바구니 확인 단계"

    # [2순위] 화면에 메뉴 영역이나 카테고리 칸이 보이는 경우 (메뉴 선택)
    elif "menu_area" in detected or "category_button" in detected:
        print("\n현재 단계: 메뉴 선택 단계")
        situation_context = "화면에 다양한 음식 메뉴들과 종류별 선택 칸이 나열되어 있어서, 먹고 싶은 음식을 손가락으로 골라 담는 단계"

    # [3순위] 버튼 글자 중에 '호출'이나 '호흡'이 들어온 경우 (직원 호출)
    elif "payment_button" in detected and any("호출" in btn or "호흡" in btn for btn in payment_buttons_text):
        print("\n현재 단계: 직원 도움 요청 단계")
        situation_context = "매장 직원의 대면 도움이 필요하여 화면에 있는 직원 호출 단추를 누른 상태"

    # [4순위] 최종 결제 수단 선택 단계 (카드, 페이, 현금 등 여러 텍스트가 섞여 들어올 때 대응)
    elif "payment_button" in detected and any(
        any(kw in btn.lower() for kw in ["카드", "페이", "현금", "코인", "캐쉬백", "pay", "zero", "point", "coupon"]) 
        for btn in payment_buttons_text
    ):
        print("\n현재 단계: 최종 결제 수단 선택 단계")
        situation_context = "주문서 작성을 모두 마치고, 신용카드나 모바일 페이 등 어떤 방법으로 돈을 낼 것인지 최종 결제 방식을 선택하는 화면 단계"

    # [5순위] 버튼은 있으나 메뉴판/장바구니가 없는 경우 (첫 화면 매장/포장)
    elif "payment_button" in detected and "menu_area" not in detected and "cart_area" not in detected:
        print("\n현재 단계: 식사 장소(매장/포장) 선택 단계")
        situation_context = "주문을 시작하기 위해 매장에서 먹고 갈 것인지, 혹은 포장해 갈 것인지 식사 장소를 선택하는 첫 화면 단계"

    # [6순위] 뒤로가기 버튼만 활성화된 경우
    elif "back_button" in detected:
        print("\n현재 단계: 이전 화면 이동 가능")
        situation_context = "이전 화면이나 처음 화면으로 돌아갈 수 있는 '취소' 또는 '뒤로가기' 버튼이 활성화된 단계"

    # [7순위] 예외 상황
    else:
        print("\n현재 단계: 알 수 없음 (안내 및 카드 삽입 지시 화면 등)")
        situation_context = "화면의 구성 요소를 명확히 파악하기 어려워 잠시 대기가 필요한 단계"

    # 🤖 GPT 가이드 문장 호출
    print("\n🤖 [GPT 가이드 문장 생성 중...]")
    ai_friendly_guide = call_friendly_gpt_guide(situation_context, payment_buttons_text)
    
    print(f"💬 약자 맞춤형 최종 안내:\n{ai_friendly_guide}")
    return ai_friendly_guide


# ==========================================
# 메인 실행 파이프라인
# ==========================================
if __name__ == "__main__":
    img = cv2.imread(str(IMAGE_PATH))

    if img is None:
        raise FileNotFoundError(f"❌ 이미지를 찾을 수 없습니다: {IMAGE_PATH}")

    results = model(str(IMAGE_PATH), conf=0.5, device=device, save=False)

    detected = set()
    payment_buttons_text = []

    print("\n=== 검출 및 OCR 결과 ===")

    for box in results[0].boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        label = results[0].names[cls]

        detected.add(label)
        print(f"\n[검출] {label} ({conf:.2f})")

        if label == "payment_button":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_btn = img[y1:y2, x1:x2]

            btn_text = extract_text_from_button(cropped_btn)

            if btn_text:
                payment_buttons_text.append(btn_text)
                print(f"        ㄴ 🔎 OCR 결과: '{btn_text}'")

    print("\n==============================")
    print("검출된 클래스 목록:", detected)
    print("결제 버튼 텍스트 모음:", payment_buttons_text)
    print("==============================")

    guide_text = generate_guide(detected, payment_buttons_text)
    print("\n[생성된 가이드 자막] " + guide_text)

    # 시각화 및 최종 저장
    h, w, _ = img.shape
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls)
        label = results[0].names[cls]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.rectangle(img, (0, h - 120), (w, h), (0, 0, 0), -1)
    cv2.putText(img, "Check Terminal for Friendly Korean Guide", (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(str(OUTPUT_PATH), img)
    print(f"\n🎯 결과 이미지 저장 완료: {OUTPUT_PATH}")
import streamlit as st
from PIL import Image
from gtts import gTTS
import os
import time

st.set_page_config(
    page_title="AI 키오스크 도우미",
    page_icon="🧾",
    layout="wide"
)

AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)


def make_tts(text):
    audio_path = os.path.join(AUDIO_DIR, "guide.mp3")

    tts = gTTS(
        text=text,
        lang="ko"
    )

    tts.save(audio_path)

    return audio_path


def fake_analysis():

    detected = [
        "cart_area",
        "payment_button"
    ]

    payment_buttons_text = [
        "결제하기"
    ]

    guide_text = """
현재 결제 단계입니다.

장바구니에 담긴 메뉴를 확인한 후
결제하기 버튼을 눌러주세요.
"""

    return detected, payment_buttons_text, guide_text


st.title("🧾 AI 키오스크 도우미")
st.markdown("---")

uploaded_file = st.file_uploader(
    "키오스크 이미지를 업로드하세요",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("업로드된 이미지")
        st.image(image, use_container_width=True)

    with col2:

        if st.button("분석 시작"):

            with st.spinner("분석 중입니다..."):
                time.sleep(1)

                detected, payment_buttons_text, guide_text = fake_analysis()

            st.success("분석 완료")

            st.subheader("현재 상황")

            if "payment_button" in detected:
                st.info("결제 단계")

            elif "menu_area" in detected:
                st.info("메뉴 선택 단계")

            else:
                st.info("상황 분석 중")

            st.subheader("안내문")
            st.write(guide_text)

            audio_path = make_tts(guide_text)

            st.subheader("음성 안내")

            with open(audio_path, "rb") as audio_file:
                st.audio(
                    audio_file.read(),
                    format="audio/mp3"
                )

            with st.expander("개발자 모드"):

                st.write("검출된 객체")
                st.write(detected)

                st.write("OCR 결과")
                st.write(payment_buttons_text)

else:
    st.info("이미지를 먼저 업로드해주세요.")
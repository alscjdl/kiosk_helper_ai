import streamlit as st
from PIL import Image
from gtts import gTTS
import os
import subprocess
from pathlib import Path

st.set_page_config(
    page_title="AI 키오스크 도우미",
    page_icon="🧾",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

IMAGE_DIR = ROOT_DIR / "images"
AUDIO_DIR = BASE_DIR / "audio"

TEST_IMAGE_PATH = IMAGE_DIR / "test.png"
RESULT_IMAGE_PATH = IMAGE_DIR / "result_guide.jpg"

IMAGE_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)


def make_tts(text):
    audio_path = AUDIO_DIR / "guide.mp3"
    tts = gTTS(text=text, lang="ko")
    tts.save(str(audio_path))
    return audio_path


def run_detect():
    result = subprocess.run(
        ["python", "detect.py"],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    return result.stdout, result.stderr, result.returncode


def extract_guide_text(stdout):
    marker = "[생성된 가이드 자막]"

    if marker in stdout:
        guide = stdout.split(marker)[-1].strip()
        guide = guide.split("\n🎯")[0].strip()
        return guide

    return "화면 분석이 완료되었습니다. 화면에 표시된 안내를 확인해 주세요."


st.title("🧾 AI 키오스크 도우미")
st.write("키오스크 화면을 분석하여 쉬운 안내문과 음성 안내를 제공합니다.")
st.markdown("---")

uploaded_file = st.file_uploader(
    "키오스크 화면 이미지를 업로드하세요.",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("먼저 키오스크 화면 이미지를 업로드해 주세요.")

else:
    image = Image.open(uploaded_file).convert("RGB")
    image.save(TEST_IMAGE_PATH)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("업로드한 이미지")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("분석 실행")
        st.write("아래 버튼을 누르면 YOLO, OCR, GPT 분석을 실행합니다.")

        if st.button("분석 시작"):
            with st.spinner("키오스크 화면을 분석하는 중입니다..."):
                stdout, stderr, returncode = run_detect()

            if returncode != 0:
                st.error("분석 중 오류가 발생했습니다.")
                st.subheader("오류 내용")
                st.code(stderr)
            else:
                st.success("분석이 완료되었습니다.")

                guide_text = extract_guide_text(stdout)

                st.subheader("📢 최종 안내문")
                st.info(guide_text)

                audio_path = make_tts(guide_text)

                st.subheader("🔊 음성 안내")
                with open(audio_path, "rb") as audio_file:
                    st.audio(audio_file.read(), format="audio/mp3")

                if RESULT_IMAGE_PATH.exists():
                    st.subheader("분석 결과 이미지")
                    st.image(str(RESULT_IMAGE_PATH), use_container_width=True)

                with st.expander("개발자 모드: detect.py 실행 로그 보기"):
                    st.code(stdout)
                    if stderr:
                        st.code(stderr)
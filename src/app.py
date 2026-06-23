import streamlit as st
from PIL import Image
from gtts import gTTS
from pathlib import Path
import subprocess

st.set_page_config(
    page_title="AI 키오스크 도우미",
    page_icon="🧾",
    layout="wide"
)

# ==========================================
# ⚠️ 경로 설정 보완
# ==========================================
BASE_DIR = Path(__file__).resolve().parent    # src/ 폴더 위치
ROOT_DIR = BASE_DIR.parent                    # 프로젝트 최상위 폴더 위치

IMAGE_DIR = ROOT_DIR / "images"
AUDIO_DIR = BASE_DIR / "audio"

# detect.py는 현재 "test.jpg"를 읽도록 하드코딩 되어 있으므로 확장자를 jpg로 일치시킵니다.
TEST_IMAGE_PATH = IMAGE_DIR / "test.jpg"

IMAGE_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)


# ==========================================
# TTS 생성
# ==========================================
def make_tts(text):
    audio_path = AUDIO_DIR / "guide.mp3"

    tts = gTTS(
        text=text,
        lang="ko"
    )

    tts.save(str(audio_path))
    return audio_path


# ==========================================
# detect.py 실행
# ==========================================
def run_detect():
    # ⚠️ app.py가 실행되는 위치에 상관없이 항상 src/detect.py가 정확하게 실행되도록 절대 경로로 처리합니다.
    detect_script_path = BASE_DIR / "detect.py"
    
    result = subprocess.run(
        ["python", str(detect_script_path)],
        cwd=str(ROOT_DIR),  # 프로젝트 최상위(ROOT) 기준으로 실행해야 .env 및 내부 경로가 꼬이지 않습니다.
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

    return result.stdout, result.stderr, result.returncode


# ==========================================
# GPT 결과 추출
# ==========================================
def extract_guide_text(stdout):
    marker = "[생성된 가이드 자막]"

    if marker in stdout:
        guide = stdout.split(marker)[-1].strip()
        # 최종 출력 포맷("\n🎯 결과 이미지 저장 완료")에 맞춤 전처리
        guide = guide.split("\n🎯")[0].strip()
        return guide

    return "화면을 분석하고 있습니다. 잠시만 기다려 주세요."


# ==========================================
# UI 스타일
# ==========================================
st.markdown("""
<style>
.main-title{
    font-size:42px;
    font-weight:800;
    margin-bottom:5px;
}
.sub-title{
    font-size:20px;
    color:#555;
    margin-bottom:25px;
}
.guide-box{
    background-color:#F0F7FF;
    border:2px solid #CDE5FF;
    border-radius:18px;
    padding:30px;
    font-size:26px;
    font-weight:600;
    line-height:1.8;
}
.step-box{
    background-color:#FFF7E6;
    border:2px solid #FFE1A8;
    border-radius:14px;
    padding:16px;
    font-size:22px;
    font-weight:700;
    margin-bottom:15px;
}
.help-box{
    font-size:18px;
    color:#555;
    line-height:1.7;
}
div.stButton > button{
    width:100%;
    height:65px;
    font-size:22px;
    font-weight:700;
    border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 헤더
# ==========================================
st.markdown(
    '<div class="main-title">🧾 AI 키오스크 도우미</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-title">키오스크 화면을 쉽게 설명하고 음성으로 안내해 드립니다.</div>',
    unsafe_allow_html=True
)
st.markdown("---")

# ==========================================
# 이미지 업로드
# ==========================================
uploaded_file = st.file_uploader(
    "키오스크 화면 사진을 선택해 주세요.",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("먼저 키오스크 화면 사진을 올려 주세요.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    image.save(TEST_IMAGE_PATH)  # 수정한 경로(jpg)로 이미지 저장

    col1, col2 = st.columns([1, 1])

    # ==========================
    # 왼쪽 영역
    # ==========================
    with col1:
        st.subheader("📷 선택한 키오스크 화면")
        st.image(
            image,
            use_container_width=True
        )

    # ==========================
    # 오른쪽 영역
    # ==========================
    with col2:
        st.subheader("🙋 안내 받기")
        st.markdown(
            """
            <div class="help-box">
            버튼을 누르면 현재 화면에서 해야 할 일을
            쉽고 친절하게 알려드립니다.
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("화면 안내 받기"):
            with st.spinner("키오스크 화면을 읽고 안내를 준비하고 있어요."):
                stdout, stderr, returncode = run_detect()

            if returncode != 0:
                st.error("화면을 확인하는 중 문제가 발생했습니다.")
                with st.expander("에러 로그 보기"):
                    st.code(stderr)
            else:
                guide_text = extract_guide_text(stdout)

                st.success("안내가 준비되었습니다.")
                st.markdown(
                    """
                    <div class="step-box">
                    📌 지금 해야 할 일
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"""
                    <div class="guide-box">
                    {guide_text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                audio_path = make_tts(guide_text)

                st.subheader("🔊 음성으로 듣기")
                with open(audio_path, "rb") as audio_file:
                    st.audio(
                        audio_file.read(),
                        format="audio/mp3"
                    )
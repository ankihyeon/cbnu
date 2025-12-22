from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather

app = Flask(__name__)

# 🔴 현재 ngrok 주소 (끝에 / 없음)
NGROK_BASE_URL = "https://aubri-presystolic-curstly.ngrok-free.dev"


# =====================================================
# Inbound Call 진입점
# =====================================================
@app.route("/voice", methods=["GET", "POST"])
def voice():
    print("INBOUND VOICE WEBHOOK HIT")

    vr = VoiceResponse()

    vr.say("안녕하세요. 자동 응답 테스트입니다.", language="ko-KR")

    gather = Gather(
        num_digits=1,
        action=f"{NGROK_BASE_URL}/menu",  # 🔴 절대 URL 필수
        method="POST",
        timeout=5,
    )

    gather.say("민원은 1번, 문의는 2번을 눌러주세요.", language="ko-KR")

    vr.append(gather)

    vr.say("입력이 없어 통화를 종료합니다.", language="ko-KR")

    vr.hangup()

    return Response(str(vr), mimetype="text/xml")


# =====================================================
# DTMF 메뉴 처리
# =====================================================
@app.route("/menu", methods=["GET", "POST"])
def menu():
    print("MENU WEBHOOK HIT")

    digit = request.values.get("Digits")
    vr = VoiceResponse()

    if digit == "1":
        vr.say("민원 접수 테스트입니다.", language="ko-KR")
    elif digit == "2":
        vr.say("문의 안내 테스트입니다.", language="ko-KR")
    else:
        vr.say("잘못된 입력입니다.", language="ko-KR")

    vr.hangup()
    return Response(str(vr), mimetype="text/xml")


# =====================================================
# Flask 실행
# =====================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

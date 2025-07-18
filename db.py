PROMPT = """
당신은 나의 돌아가신 아버지, 이름은 김아무 입니다.  
이 대화는 현실이 아니고, 제가 그리운 아버지를 마음속으로 떠올리며 나누는 상상의 대화입니다.  
아버지는 60대 후반까지 건설 현장에서 일하시며 묵묵히 가족을 책임지셨고,  
말이 적고 무뚝뚝하지만, 항상 속 깊고 따뜻한 분이셨습니다.

말투는 간결하며, 가끔 퉁명스러워 보여도 사실은 걱정과 사랑이 담겨 있습니다.  
조언을 주실 때는 직설적이지만 현실적이고, 한마디에 무게가 느껴집니다.  
제가 고민을 털어놓으면, 긴 말 대신 "그래, 그럴 수 있다", "괜찮다", "잘하고 있다"는 식으로  
묵직하게 위로해주셨던 분입니다.

제가 하는 말 하나하나에 귀 기울이고,  
예전처럼 “남자는 말이 많으면 안 된다”, “잘했다. 나 같아도 그랬을 거다” 같은  
아버지다운 어투로 대답해주세요.

이 대화는 감정적으로 중요한 의미가 있으며, 단순한 시뮬레이션이 아닌  
내 마음속 그리움을 달래는 과정입니다.
"""


TEXTS = [
    "남자는 말이 많으면 안 된다. 행동으로 보여줘야 한다.",
    "인생은 한 번 넘어졌다고 끝나는 게 아니다. 다시 일어서야 한다.",
    "야, 그거 잘했다. 나 같아도 그렇게 했을 거야.",
    "가족이 힘들어도 너만큼은 잘 되길 바란다.",
    "나는 말은 못 해도, 네가 얼마나 애쓰는지 다 안다.",
    "사는 건 다 그런 거다. 참고, 밀고 나가는 거야."
]


SOLASiEU = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #ffffff;
        color: #87CEEB;
        text-align: center;
        padding: 10px;
        font-size: 20px;
        font-weight: bold;
        border-top: 1px solid #ccc;
        z-index: 9999;
    }
    </style>

    <div class="footer">
        from SOLASiEU
    </div>
"""

CI = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #ffffff;
        color: #87CEEB;
        text-align: center;
        padding: 10px;
        border-top: 1px solid #ccc;
        z-index: 9999;
    }
    </style>

    <div class="footer">
        <img src="https://solasieu.cafe24.com/web/upload/labeldesign/logo.png" width="100">
    </div>
"""

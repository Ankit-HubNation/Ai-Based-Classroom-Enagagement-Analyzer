def calculate_engagement(emotion, looking_forward):
    emotion_score = {
    "Happy": 0.95,
    "Surprise": 0.85,
    "Neutral": 0.7,
    "Fear": 0.5,
    "Sad": 0.4,
    "Angry": 0.3,
    "Disgust": 0.2
}

    attention_score = 1 if looking_forward else 0.3

    return round((emotion_score.get(emotion, 0.5) * 0.6 +
                  attention_score * 0.4) * 100, 2)
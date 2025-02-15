import json
import requests

def emotion_detector(text_to_analyze):
    """
    Calls the Watson Emotion Predict API with the provided text,
    extracts emotion scores from the response, determines the dominant emotion,
    and returns a dictionary with the scores and the dominant emotion.
    
    If the API returns a status code 400 (bad request), returns a dictionary with all values as None.
    
    Args:
        text_to_analyze (str): The text to analyze for emotion.
    
    Returns:
        dict: A dictionary in the format:
            {
                'anger': anger_score,
                'disgust': disgust_score,
                'fear': fear_score,
                'joy': joy_score,
                'sadness': sadness_score,
                'dominant_emotion': '<name of the dominant emotion>' or None if error.
            }
    """
    # API endpoint and headers as provided.
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    payload = {"raw_document": {"text": text_to_analyze}}

    # Make the POST request.
    response = requests.post(url, headers=headers, json=payload)
    
    # If the API returns a 400 status code, return dictionary with all values as None.
    if response.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None
        }
    
    response.raise_for_status()
    
    # Parse the JSON response from the API.
    response_data = response.json()
    
    # Debug: Print the full response to inspect its structure.
    print("Full API response:", response_data)
    
    # Extract emotion results depending on the response structure.
    if "text" in response_data:
        try:
            emotion_results = json.loads(response_data["text"])
        except json.JSONDecodeError as e:
            raise ValueError("Failed to decode JSON from 'text' attribute.") from e
    elif "emotionPredictions" in response_data:
        predictions = response_data.get("emotionPredictions", [])
        if predictions and "emotion" in predictions[0]:
            emotion_results = predictions[0]["emotion"]
        else:
            raise ValueError("Could not extract emotion predictions from the response.")
    else:
        raise ValueError("Unexpected response format from API.")
    
    # Extract the required emotion scores, defaulting to 0 if not found.
    anger_score   = emotion_results.get("anger", 0)
    disgust_score = emotion_results.get("disgust", 0)
    fear_score    = emotion_results.get("fear", 0)
    joy_score     = emotion_results.get("joy", 0)
    sadness_score = emotion_results.get("sadness", 0)
    
    # Create a dictionary of the emotion scores.
    emotion_scores = {
        "anger": anger_score,
        "disgust": disgust_score,
        "fear": fear_score,
        "joy": joy_score,
        "sadness": sadness_score
    }
    
    # Determine the dominant emotion (the one with the highest score).
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    
    # Return the formatted result.
    return {
        "anger": anger_score,
        "disgust": disgust_score,
        "fear": fear_score,
        "joy": joy_score,
        "sadness": sadness_score,
        "dominant_emotion": dominant_emotion
    }

"""
Server module for NLP Emotion Detection.

This module creates a Flask application that provides a web interface for analyzing
emotions from text input using the Watson NLP API. It includes routes for displaying
the form and processing emotion detection requests.
"""

from flask import Flask, render_template, request
from EmotionDetection import emotion_detector

app = Flask(__name__)


@app.route('/')
def index():
    """
    Render the index page.

    Returns:
        str: Rendered HTML template for the index page.
    """
    return render_template('index.html')


@app.route('/emotionDetector', methods=['GET', 'POST'])
def emotion_detection_view():
    """
    Process emotion detection requests.

    For POST requests, retrieves the statement from the form, analyzes it using the
    emotion_detector function, and renders the result. If the input is blank or the
    API returns a 400 (resulting in a None dominant emotion), an error message is shown.

    Returns:
        str: Rendered HTML template with either the analysis result or an error message.
    """
    if request.method == 'POST':
        statement = request.form.get('statement')
        if not statement:
            return render_template('index.html', result="Please enter a valid statement.")

        result = emotion_detector(statement)

        # If dominant_emotion is None, display an error message.
        if result.get('dominant_emotion') is None:
            return render_template('index.html', result="Invalid text! Please try again!")

        response_string = (
            f"For the given statement, the system response is 'anger': {result.get('anger')}, "
            f"'disgust': {result.get('disgust')}, 'fear': {result.get('fear')}, "
            f"'joy': {result.get('joy')} and 'sadness': {result.get('sadness')}. "
            f"The dominant emotion is {result.get('dominant_emotion')}."
        )
        return render_template('index.html', result=response_string)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

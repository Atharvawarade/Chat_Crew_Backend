import speech_recognition as sr

def get_voice_input():
    """
    Captures voice input from the user and converts it to text.
    Returns the recognized text or an error message if recognition fails.
    """
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    print("Listening... Please speak into the microphone.")
    
    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5)
            
        # Convert the audio to text
        recognized_text = recognizer.recognize_google(audio)
        print(f"You said: {recognized_text}")
        return recognized_text
    except sr.UnknownValueError:
        print("Sorry, I could not understand your voice.")
        return "Error: Could not understand your voice."
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return "Error: Issue with the recognition service."
    except sr.WaitTimeoutError:
        print("Listening timed out.")
        return "Error: Listening timed out."

# Example usage
if __name__ == '__main__':  # Corrected line
    query = get_voice_input()
    if "Error" not in query:
        print(f"Processing query: {query}")
    else:
        print("Try speaking again.")
 

import speech_recognition as sr

def speech_to_text(audio):
    r = sr.Recognizer()
    with sr.AudioFile(audio) as source:
        audio_data = r.record(source)
    try:
        return r.recognize_google(audio_data)
    except:
        return "Audio not clear"

def record_and_recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!") # This will print to the console where Streamlit is running
        r.adjust_for_ambient_noise(source) # Adjusts for ambient noise
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print("You said: " + text)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return "Audio not clear"
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return "Speech recognition service error"
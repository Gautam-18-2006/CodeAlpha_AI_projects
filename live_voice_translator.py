import speech_recognition as sr
from googletrans import Translator
import asyncio
import edge_tts
import playsound
import os
import uuid

translator = Translator()
async def speak(text):
    filename = f"{uuid.uuid4()}.mp3"
    communicate = edge_tts.Communicate(
        text=text,
        voice="en-US-GuyNeural"
    )
    await communicate.save(filename)
    playsound.playsound(filename, True)
    os.remove(filename)

print("🎤 Speak Hindi (Ctrl+C to stop)")

while True:
    r = sr.Recognizer()
    r.pause_threshold = 0.5

    try:
        with sr.Microphone() as source:
            print("\nListening...")
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio = r.listen(source, phrase_time_limit=4)

        hindi_text = r.recognize_google(audio, language="hi-IN")
        print("Hindi:", hindi_text)

        if len(hindi_text) > 80:
            hindi_text = hindi_text[:80]

        translated = translator.translate(hindi_text, src="hi", dest="en")
        english_text = translated.text
        print("English:", english_text)

        asyncio.run(speak(english_text))

    except sr.UnknownValueError:
        print("❌ Could not understand")
    except KeyboardInterrupt:
        print("\nStopped")
        break

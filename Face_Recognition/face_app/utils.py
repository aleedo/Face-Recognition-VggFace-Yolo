import os
import random

def saying_sorry():
    answers = [
        "No Face Detected: اختار صورة فيها وش يا أمجد",
        "No Face Detected: Why did the photographer get fired? Because they couldn't face the fact that there were no faces in the photos.",
        "No Face Detected: Why did the computer say there were no faces in the photo? Because it was too busy Facebooking!",
        "No Face Detected: Why did the selfie fail? Because there was no one in the photo to take a selfie with!",
        "No Face Detected: What do you call a photo with no faces? A faceless photo!",
        "No Face Detected: Why did the faceless photo get rejected from the art gallery? Because it lacked character!",
        "No Face Detected: Why did the detective refuse to investigate the photo? Because it was a case of missing faces!",
        "No Face Detected: Why did the magician refuse to perform for the photo? Because he couldn't make the faces disappear any further!",
        "No Face Detected: Why did the comedian make fun of the photo? Because there was no one there to laugh at his jokes!",
        "No Face Detected: Why did the makeup artist refuse to work on the photo? Because there were no faces to apply makeup on!",
        "No Face Detected: Why did the photo get a bad grade? Because it was missing the most important part - a face!",
    ]

    weights = [0.5] + [0.5 / (len(answers) - 1)] * (len(answers) - 1)
    choice = random.choices(answers, weights=weights)[0]
    return choice

def sorry_img():
    img_paths = [
    'https://shorturl.at/oAQ05',
    'https://shorturl.at/frt14',
    'https://shorturl.at/lnrBZ',
    'https://shorturl.at/kqxV1',
    'https://shorturl.at/stvQY',
    'https://shorturl.at/esMQ6',
    'https://shorturl.at/kJNT3',
    'https://shorturl.at/egovS',
    'https://shorturl.at/xKTUV',
    'https://shorturl.at/tuMW6',
    ]
    return random.choice(img_paths)

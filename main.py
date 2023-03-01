# ----------------> lipraries
import pyttsx3
import speech_recognition as sr
import sys
import json
from vosk import Model, KaldiRecognizer
import pyaudio
import face_recognition
import os
import io
import json
import random
from time import sleep
import cv2
import numpy as np
import pytesseract
import requests
from PIL import Image
from gtts import gTTS
from textblob import TextBlob
import spacy






#-------------------> speak function
def speaktext(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()



##------------------offline speach to text
model = Model(r"C:\Users\hp\PycharmProjects\pythonProject\vosk-model-small-en-us-0.15") #----> loading model
recognizer = KaldiRecognizer(model, 16000)
mic = pyaudio.PyAudio()
stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
stream.start_stream()



speaktext(" Hallo sir ")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    speaktext("Cannot open video")
    raise IOError("Cannot open video")


class face_reconizer_train:
    def __init__(self):
        path = 'pics'
        images = []
        classnames = []
        mylist = os.listdir(path)
        print(mylist)

        for cl in mylist:
            cur = cv2.imread(f'{path}/{cl}')
            images.append(cur)
            classnames.append(os.path.splitext(cl)[0])

        print(classnames)
        # open file
        with open('names.txt', 'w+') as f:
            # write elements of list
            for items in classnames:
                f.write('%s\n' % items)
            print(classnames)
            print("File1 written successfully")

        # close the file
        f.close()

        def findencode(images):
            encodelist = []
            for img in images:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encode = face_recognition.face_encodings(img)[0]
                encodelist.append(encode)
            return encodelist

        encodelist = findencode(images)
        # open file
        file = open("values.txt", "r+")
        file.truncate(0)
        file.close()
        with open('values.txt', 'w+') as f:

            # write elements of list
            for items in encodelist:
                for i in items:
                    f.write('%f\n' % i)
            print("File2 written successfully")

        # close the file
        f.close()
        print("encoding complete")


class face_reconizer_main:
    def __init__(self):
        encodelist = []
        arr = []
        c = 0
        i = 0
        with open('values.txt', 'r+') as f:
            for item in f:
                c += 1
                n = float(item)
                arr.append(n)
                if (c == 128):
                    #   print(arr)
                    encodelist.append(arr)
                    arr = []
                    c = 0
            print("File readed successfully")
        # close the file
        f.close()
        # print(encodelist)
        # open file
        classnames = []
        with open('names.txt', 'r+') as f:
            # write elements of list
            for items in f:
                classnames.append(str(items))
            print("File readed successfully")
        print(classnames)
        # close the file
        f.close()
        # ----------------------> open camera

        name1 = " "
        ##-------------------> search and reconize faces
        bo=1
        while bo:
        ##----------------------offline speach to text
            data = stream.read(4096, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                myText = recognizer.Result()
                myText = myText[14:-3]
                myText = myText.lower()
                print(myText)
                if ('end' in myText or 'close' in myText or 'exit' in myText):
                    speaktext("  ok   sir ")
                    print("ok sir")

                    cap.release()
                    bo=0
                    break


            succes, img = cap.read()
            imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

            cur_faceloc = face_recognition.face_locations(imgs)
            cur_encode = face_recognition.face_encodings(imgs, cur_faceloc)

            for encode_face, faceloc in zip(cur_encode, cur_faceloc):
                match = face_recognition.compare_faces(encodelist, encode_face)
                faced = face_recognition.face_distance(encodelist, encode_face)
                print(faced)
                matchindx = np.argmin(faced)

                if match[matchindx]:
                    name = classnames[matchindx].upper()
                    print("name", name)
                    y1, x2, y2, x1 = faceloc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                    # ----> saying
                    if (name1 != name):
                        name1 = name
                        print("name11111", name1)
                        speaktext(name1)

                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0))
                    cv2.putText(img, "dont know broo", (x1 - 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, fontScale=1,
                                color=(0, 255, 0), thickness=3)
                    speaktext("dont know broo")


            cv2.imshow("wee", img)
            key = cv2.waitKey(5) & 0xFF
            if key == ord("x"):
                print("Turning off camera.3333")
                print("Camera off3333333333.")
                print("Program ended33333333333.")
                cap.release()
                cv2.destroyAllWindows()
                bo=0


class object_detection_main:
    def __init__(self):
        config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        frozen_model = 'frozen_inference_graph.pb'


        model2 = cv2.dnn_DetectionModel(frozen_model, config_file)
        classlabel = []
        file_name = 'label.txt'
        with open(file_name, 'rt') as fpt:
            classlabel = fpt.read().rstrip('\n').split('\n')

        model2.setInputSize(320, 320)
        model2.setInputScale((1.0 / 127.5))
        model2.setInputMean((127.5, 127.5, 127.5))
        model2.setInputSwapRB(1)

        font_scale = 3
        font = cv2.FONT_HERSHEY_PLAIN
        name = ""

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            speaktext("Cannot open video")
            raise IOError("Cannot open video")

        bo=1
        name1 = "1"
        while (bo):
            try:
                data = stream.read(4096, exception_on_overflow=False)
                if recognizer.AcceptWaveform(data):
                    myText = recognizer.Result()
                    myText = myText[14:-3]
                    myText = myText.lower()
                    print(myText)
                    if ('end' in myText or 'close' in myText or 'exit' in myText):
                        speaktext("  ok   sir ")
                        print("ok sir")
                        bo=0
                        break

                ret, frame = cap.read()
                classin, conf, bbox = model2.detect(frame, confThreshold=0.6)

                print(classin)
                if (len(classin) != 0):
                    for ClassInd, conf, boxes in zip(classin.flatten(), conf.flatten(), bbox):
                        if (ClassInd <= 80):
                            cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                            cv2.putText(frame, classlabel[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font,
                                        fontScale=font_scale,
                                        color=(0, 255, 0), thickness=3)
                            # saying
                            name = classlabel[ClassInd - 1]
                            if (name1 != name):
                                name1 = name
                                print("name here", name1)
                                speaktext(name1)
                        else:
                            cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                            cv2.putText(frame, "dont know broo", (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale,
                                        color=(0, 255, 0), thickness=3)

                    cv2.imshow('Object Detection Tutorial', frame)


                key = cv2.waitKey(5) & 0xFF
                if key == ord("x"):
                    print("Turning off camera.3333")
                    print("Camera off3333333333.")
                    print("Program ended33333333333.")
                    cap.release()
                    cv2.destroyAllWindows()
                    bo=0

            except KeyboardInterrupt:
                print("Turning off camera.")
                print("Camera off.")
                print("Program ended.")
                bo=0
                cap.release()
                cv2.destroyAllWindows()




class ocr_main:
    def __init__(self):
        # ---------- image from camera

        key = cv2.waitKey(1)
        webcam = cv2.VideoCapture(0)
        sleep(2)
        speaktext("For Recognize Image PRESS   ")


        print("For Recognize Image PRESS 'S'\n"
              "For QUIT PRESS 'Q\n"
              "After run time if 'images.jpg' is still visible,Please re-run the program.\n")
        while True:
            try:
                data = stream.read(4096, exception_on_overflow=False)
                if recognizer.AcceptWaveform(data):
                    myText =recognizer.Result()
                    myText = myText[14:-3]
                    myText = myText.lower()
                    print(myText)

                    if ('end' in myText or 'close' in myText or 'exit' in myText):
                        speaktext("  ok   sir ")
                        print("ok sir")
                        speaktext("buy  buy")
                        break


                check, frame = webcam.read()
                print(check)  # prints true as long as the webcam is running
                print(frame)  # prints matrix values of each framecd
                cv2.imshow("Capturing", frame)


                key = cv2.waitKey(5) & 0xFF
                if key == ord('s'):
                    cv2.imwrite(filename='images.jpg', img=frame)
                    r = random.randint(1, 20000000)
                    img_file = 'images' + str(r) + '.jpg'
                    cv2.imwrite(filename='data/' + img_file, img=frame)
                    webcam.release()
                    print("Processing image...")
                    img = cv2.imread('images.jpg', cv2.IMREAD_ANYCOLOR)
                    print("Image saved!")
                    cv2.destroyAllWindows()
                    break


                if key == ord("x"):
                    print("Turning off camera.3333")
                    print("Camera off3333333333.")
                    print("Program ended33333333333.")
                    webcam.release()
                    cv2.destroyAllWindows()
                    break


            except KeyboardInterrupt:
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                break



            key = cv2.waitKey(5) & 0xFF
            if key == ord("x"):
                print("Turning off camera.3333")
                print("Camera off3333333333.")
                print("Program ended33333333333.")
                webcam.release()
                cv2.destroyAllWindows()
                break
        sleep(2)
        #
        # #-------- saved image
        # resim = "Capture1.PNG"
        # img = cv2.imread(resim)
        # print("Picture is Detected")

        # ------------- edit emage
        img = frame
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('aaaa', img)


        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        img = Image.fromarray(img)

        text = pytesseract.image_to_string(img)
        file = open('text.txt', 'w')
        file.write(text)
        file.close()
        print(text)
        # ---------------correct the spelling

        text = TextBlob(text)
        result = text.correct()
        print(result)

        '''
        spell = SpellChecker()
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        misspelled = spell.unknown(doc)
        result=""
        result1=""
        for word in misspelled:
            # Get the one `most likely` answer
            print(spell.correction(word))
            result=result+(spell.correction(word))+" "

            # Get a list of `likely` options
            print(spell.candidates(word))
            result1 = result1 + (spell.candidates(word)) + " "
            '''
        #  ------------------------- online # convert this text to speech

        # language = 'ar'  # 'en'
        # audio = gTTS(text=text, lang=language, slow=False)
        # audio.save("example.mp3")
        # os.system("start example.mp3")

        #  ------------------------- offline  convert this text to speech

        # setting new voice rate (faster) ordnary is 200
        speaktext(result)


# =---------------------------> start to take the vioce orders from user
bol = 1
x = -1
while (bol):
    data = stream.read(4096, exception_on_overflow=False)
    if recognizer.AcceptWaveform(data):
        myText = recognizer.Result()
        myText = myText[14:-3]
        myText = myText.lower()
        print(myText)
        if ('face' in myText or 'fece' in myText or 'fc' in myText or 'fees' in myText or 'face recognition' in myText or 'face reconition' in myText or 'facer' in myText or 'see' in myText or 'faces' in myText or 'who' in myText):
            speaktext("opening face reconizer ")
            face_reconizer_main()
            myText=""
            speaktext("closing face reconizer ")


        elif ('object' in myText or 'object detection' in myText or 'detect object' in myText or 'where' in myText or 'car' in myText or 'what' in myText):

            speaktext("opening object detection ")
            object_detection_main()
            myText = ""
            speaktext("closing object detection ")


        elif ('ocr' in myText or 'o c r' in myText or 'read' in myText or 'reading' in myText or 'book' in myText or 'paper' in myText or 'newspaper' in myText or 'text' in myText):

            speaktext("opening  o c r  ")
            ocr_main()
            myText = ""
            speaktext("closing o c r ")


        elif ('end' in myText or 'close' in myText or 'exit' in myText):
            bol = 0
            speaktext("  ok   sir ")
            print("ok ok sir")
            speaktext("buy  buy")
import cv2
import dlib
import os
#import sys
#import math
import time
import sqlite3
#from PIL import Image
import openface.openface.align_dlib as openface

cam = cv2.VideoCapture(-1)
detector = dlib.get_frontal_face_detector()
date = time.strftime("%d_%m_%Y")
col=['09_00__09_50','09_50__10_40','10_40__11_50','12_30__01_15','01_15__01_50','02_20__04_10','04_10__05_50']
path = './recognized_images/' + date
dlibFacePredictor = 'shape_predictor_68_face_landmarks.dat'
align = openface.AlignDlib(dlibFacePredictor)
if not os.path.exists(path):
    os.makedirs(path)



def getTimeSlot():
    date = time.strftime("%I_%M")
    for i in col:
        if int(date[0:2])==int(i[0:2]) and int(date[0:2])==int(i[7:9]) and int(date[3:5])>=int(i[3:5]) and int(date[3:5])<=int(i[10:12]):
            return i
        elif int(date[0:2])>=int(i[0:2]) and int(date[0:2])<=int(i[7:9]) and (int(date[3:5])>=int(i[3:5]) or int(date[3:5])<=int(i[10:12])):
            return i    
      
    


def getProfile(id):
    connect = sqlite3.connect("Face-DataBase")
    cmd = "SELECT * FROM s"+str(date) +" WHERE ID=" + str(id)
    cursor = connect.execute(cmd)
    profile = None
    for row in cursor:
        slot=getTimeSlot()
        profile = row
        cmd1=f"UPDATE s{str(date)} SET p{str(slot)} = 1 WHERE ID = {str(row[0])} "
        connect.execute(cmd1)
    connect.commit()
    connect.close()
    return profile


def createTable():
    connect = sqlite3.connect("Face-DataBase")
    cmd = "CREATE TABLE IF NOT EXISTS s"+str(date) +" AS SELECT * FROM Students ORDER BY Roll"
    connect.execute(cmd)
    for i in col:
        cmd="ALTER TABLE s"+str(date) +" ADD p"+str(i)+" int"
        connect.execute(cmd)
    
    connect.close()
    return

def recognize():
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read('./recognizer/trainingData.yml')
    
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255, 255, 255)
    create = input('do you want to create new DB for today(y/n) : ')
    if create.upper()=='Y':
        createTable()
    
    picNum = time.strftime("%H.%M.%S")
    img = cv2.imread('testing_images/pic23.jpg')
 #   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(img, 1)
    totalConfidence = 0.0
    faceRecognized = 0
    for i, d in enumerate(dets):
        img2 = img[d.top():d.bottom(), d.left():d.right()]
        rgbImg = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
  #      bb = align.getLargestFaceBoundingBox(rgbImg)
        alignedFace = align.align(96, rgbImg, bb=None, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        id, conf = rec.predict(alignedFace)
    
        if conf > 50 and conf < 101:
            totalConfidence += conf
            faceRecognized += 1
            profile = getProfile(id)
    
            if profile != None:
                cv2.putText(img,
                            profile[1] + str("(%.2f)" % conf),
                            (d.left(), d.bottom()),
                            fontFace, fontScale, fontColor,
                            )
        else:
            cv2.putText(img,
                        "Unknown" + str(conf),
                        (d.left(), d.bottom()),
                        fontFace, fontScale, fontColor,
                        )
    
        cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 255, 255), 2)
    cv2.imwrite(path + '/pic' + str(picNum) + '.jpg', img)
    detectPrint = 'Frame' + str(picNum) + ". %d face detected" % len(dets)
    
    if faceRecognized != 0:
        print(f"{detectPrint} and  {faceRecognized} face recognized with confidence {(totalConfidence / faceRecognized)}")
    else:
        print(f"{detectPrint} and 0 faces recognized")
    
    cv2.destroyAllWindows()



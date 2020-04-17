import cv2
import numpy as np
import sqlite3
import dlib
import os

cap = cv2.VideoCapture(-1)
detector = dlib.get_frontal_face_detector()

def insertOrUpdate(Id, Name, roll):
    connect = sqlite3.connect("Face-DataBase")
    cmd = "SELECT * FROM Students WHERE ID = " + Id
    cursor = connect.execute(cmd)
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
    if isRecordExist == 1:
        connect.execute("UPDATE Students SET Name = ? WHERE ID = ?", (Name, Id))
        connect.execute("UPDATE Students SET Roll = ? WHERE ID = ?", (roll, Id))
    else:
        params = (Id, Name, roll)
        connect.execute("INSERT INTO Students VALUES(?, ?, ?)", params)
    connect.commit()
    connect.close()


 
def enroll(pid,pname,proll):

    Id = pid
    name = pname
    roll = proll
    insertOrUpdate(Id, name, roll)


    folderName = "user" + Id
    folderPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "image_dataset/"+folderName)
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    
    sampleNum = 0
    while(True):
        ret, img = cap.read()
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detector(img, 1)
        for i, d in enumerate(dets):
            sampleNum += 1
            cv2.imwrite(folderPath + "/User." + Id + "." + str(sampleNum) + ".jpg",
                        img[d.top():d.bottom(), d.left():d.right()])
            cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)
            cv2.waitKey(200)
        cv2.imshow('frame', img)
        if(sampleNum >= 20):
            break

    cap.release()
    cv2.destroyAllWindows()



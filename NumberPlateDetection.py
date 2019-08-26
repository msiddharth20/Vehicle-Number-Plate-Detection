import cv2
import pytesseract
import numpy as np
from matplotlib import pyplot as plt

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'

plate_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

img = cv2.imread("C:\\Users\\Kunal\\PycharmProjects\\VehicleNumberPlateDetection\\vehicel_number_plate\\plate2.jpg")
# 3, 4, 5, {6, 7, 8, 9, 10, 11}
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cropped_image = img
count = 0

def image_to_text(crop_image):
    global count
    text = pytesseract.image_to_string(crop_image, lang='eng')
    str = ""
    for i in text:
        if i.isalnum():
            str += i
            count += 1
    return str


def method1():

    global img, gray_img, plate_cascade, cropped_image, count

    plate = plate_cascade.detectMultiScale(gray_img, scaleFactor=1.02, minNeighbors=5)

    flag = 0
    for x,y,w,h in plate:
        cropped_image = img[y:y + h, x:x + w]
        flag = 1
        break

    if flag == 1:
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        ret, cropped_image = cv2.threshold(cropped_image, 127, 255, cv2.THRESH_BINARY)
        text = image_to_text(cropped_image)
        # cv2.imshow("1st method", cropped_image)
        # cv2.waitKey(0)
        if text == "":
            count = 0
            method2()
        else:
            if count >= 9:
                print(text)
            else:
                ret, cropped_image = cv2.threshold(cropped_image, 127, 255, cv2.THRESH_BINARY_INV)
                text = image_to_text(cropped_image)
                if text == "":
                    count = 0
                    method2()
                else:
                    print(text)
    else:
        method2()


def method2():

    global gray_img, img, cropped_image, count

    gray = cv2.bilateralFilter(gray_img, 11, 17, 17)
    # cv2.imshow("Gray Filter", gray)

    edged = cv2.Canny(gray, 170, 200)
    # cv2.imshow("Edged", edged)

    cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img1 = img.copy()
    cv2.drawContours(img1, cnts, -1, (0, 255, 0), 3)
    # cv2.imshow("all contours", img1)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    NumberPlateCnt = None

    img2 = img.copy()
    cv2.drawContours(img2, cnts, -1, (0, 255, 0), 3)
    # cv2.imshow("all contours above 30", img2)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            x, y, w, h = cv2.boundingRect(c)
            cropped_image = img[y:y + h, x:x + w]
            break

    # cv2.drawContours(img, [NumberPlateCnt], -1, (0, 255, 0), 3)
    # cv2.imshow("Final Image", cropped_image)
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    ret, cropped_image = cv2.threshold(cropped_image, 127, 255, cv2.THRESH_BINARY)
    # ret, cropped_image = cv2.threshold(cropped_image, 127, 255, cv2.THRESH_BINARY_INV)
    text = image_to_text(cropped_image)
    if count >= 10:
        print(text)
    else:
        count = 0
        ret, cropped_image = cv2.threshold(cropped_image, 127, 255, cv2.THRESH_BINARY_INV)
        text = image_to_text(cropped_image)
        if text == "":
            pass
        else:
            print(text)


method1()
cv2.imshow("Gray", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
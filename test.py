import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'

img = cv2.imread("C:\\Users\\Kunal\\PycharmProjects\\VehicleNumberPlateDetection\\vehicel_number_plate\\plate11.jpg")

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.bilateralFilter(gray_img, 11, 17, 17)
# cv2.imshow("Gray Filter", gray)

edged = cv2.Canny(gray, 170, 200)
# cv2.imshow("Edged", edged)

cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

img1 = img.copy()
cv2.drawContours(img1, cnts, -1, (0, 255, 0), 3)
# cv2.imshow("all contours", img1)


cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:40]
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

cv2.drawContours(img, [NumberPlateCnt], -1, (0, 255, 0), 3)
cv2.imshow("Final Image", img)

cv2.imshow("Cropped Image", cropped_image)
text = pytesseract.image_to_string(cropped_image, lang='eng')
for i in text:
    if i.isalnum():
        print(i, end="")
cv2.waitKey(0)



cv2.destroyAllWindows()
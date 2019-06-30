import cv2 as cv
import os
import sys
import numpy as np
import requests as rq
from bs4 import BeautifulSoup
import json


WIDTH = 800
TAGS = "computer+solo"
#BASE_URL = "http://104.24.110.222/" #http://danbooru.idanbooru.com
BASE_URL = "http://danbooru.idanbooru.com"
START_URL = BASE_URL + "/posts?ms=1&page={}&tags={}&utf8=%E2%9C%93"
PAGES = 5
PADDING = 1.5
MIN_AREA = 1000

SIZE = 512

DIRTY_TAGS = []

face_cascade = cv.CascadeClassifier("lbpcascade_animeface.xml")

def crop_face(image):
    faces = face_cascade.detectMultiScale(image, 1.2, 5)

    h0, w0 = image.shape[:2]

    if w0 > h0:
        w = h0
        h = h0
        y = h // 2
        x = (w0 - w) // 2 + w // 2
    else:
        h = w0
        w = w0
        x = w // 2
        y = h // 2

    if (len(faces) != 0):
        faces = sorted(faces, key=(lambda f: f[2] * f[3]), reverse=True)
        x1, y1, w1, h1 = faces[0]
        if (w1 * h1 >= MIN_AREA):
            x = x1 + w1 // 2
            y = y1 + h1 // 2

    # x -= int(w * (PADDING-1) / 2)
    # y -= int(h * (PADDING-1) / 2)
    #
    # w = int(w * PADDING)
    # h = int(h * PADDING)

    if (x < w // 2):
        x = w // 2
    if (y < h // 2):
        y = h // 2
    if x > w0 - w // 2:
        x = w0 - w // 2
    if y > h0 - h // 2:
        y = h0 - h // 2

    x -= w // 2
    y -= h // 2

    face = image[y:y + h, x:x + w]
    face = cv.resize(face, (SIZE, SIZE))
    return face

srcs = []
for page in range(1,PAGES):
    url = START_URL.format(page, TAGS)
    START = rq.get(url)
    parsed = BeautifulSoup(START.text)

    imgs = parsed.body.find_all('a')
    for img in imgs:
        if img.find('picture') == None:
            continue
        if img.has_key("href") and "posts" in img["href"]:
            srcs.append(img["href"])


for src in srcs:
    res = rq.get(BASE_URL + src)
    parsed = BeautifulSoup(res.text)
    imgs = parsed.find_all("img")
    if len(imgs) == 0:
        continue
    img = imgs[0]
    if not img.has_attr("src"):
        continue

    if img.has_attr("class"):
        continue

    src = img["src"]
    tags = img["data-tags"]
    tags = tags.split(" ")

    print(tags)

    br = False
    for tag in DIRTY_TAGS:
        if tag in tags:
            br = True
            break
    if br:
        continue

    resp = rq.get(src)
    image = np.asarray(bytearray(resp.content), dtype="uint8")
    image = cv.imdecode(image, cv.IMREAD_COLOR)

    face = crop_face(image)
    if face is None:
        continue

    cv.imshow("Image", image)
    cv.imshow("Face", face)
    cv.waitKey(10)



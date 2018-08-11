import base64
import http.client
import json
import os
import ssl
from PIL import Image
from collections import namedtuple
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
from Nešto import Run

headers = {"Content-type": "application/json",
           "X-Access-Token": "iaKSxjU2MA754zoskX6BCXXPRDwolazDDAKv"}
conn = http.client.HTTPSConnection("dev.sighthoundapi.com", 
       context=ssl.SSLContext(ssl.PROTOCOL_TLSv1))


image_data = "http://www.automanija.com/wp-content/uploads/2012/09/Hrvatski-Auto-Festival-2012-retrospektiva-18.jpg"

# To use a local file uncomment the following line and update the path
#image_data = base64.b64encode(open("/path/to/local/image.jpg").read())

params = json.dumps({"image": image_data})
conn.request("POST", "/v1/recognition?objectType=licenseplate", params, headers)
response = conn.getresponse()
result = response.read()
z = json.loads(result, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
print (z.objects[0].licenseplateAnnotation.bounding.vertices)
q = z.objects[0].licenseplateAnnotation.bounding.vertices
img = Image.open("Hrvatski-Auto-Festival-2012-retrospektiva-18.jpg")
area = (q[0].x, q[0].y, q[2].x, q[2].y)
cropped_img = img.crop(area)
cropped_img.show()
w = z.objects[0].licenseplateAnnotation.attributes.system.characters
os.mkdir('./plate/')
cropped_img_list = list();
for i in range(0, len(w)):
    print(w[i].bounding.vertices)
    l = w[i].bounding.vertices
    area = (l[0].x, l[0].y, l[2].x, l[2].y)
    cropped_img_list.append(img.crop(area))
    cropped_img_list[i].save('./plate/'+ str(i) + '.jpg')

    
Run()
cropped_img.save('img.jpg')



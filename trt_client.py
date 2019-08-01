import requests
import cv2

# bytes_img_path = "/home/clliao/圖片/binary_image"
image_path = "/home/clliao/圖片/miku-100x100.jpg"
image = cv2.imread(image_path)
array_img = image.astype("float32")
bytes_img = array_img.tobytes()
batch_size = 4

headers = {
    "NV-InferRequest": 'batch_size: %d input { name: "input_1" dims: 100 dims: 100 dims: 3} output { name: "output_1"  cls { count: 3 } }' % batch_size,
}

url = "http://localhost:8000/api/infer/test_model"

response = requests.request("POST", url, data=bytes_img*batch_size, headers=headers)
# response = requests.request("POST", url, data=open(bytes_img_path, 'rb'), headers=headers)

print(response.text)

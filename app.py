"""
Run a rest API exposing the yolov5s object detection model
"""


from glob import glob
import random
import io
from PIL import Image
import json

import torch
from flask import Flask, request, render_template,send_file,url_for
from flask_cors import CORS, cross_origin
import base64
import random
import requests
from pathlib import Path
import datetime
import io
from werkzeug.utils import secure_filename
from io import BytesIO
import time
# from segmentation.segment.predict import run



# derain import
from derain_dehaze.inference import pred_detection


# segmenation
# from segmentation import segment_image


UPLOAD_FOLDER = 'data'
PREDECTION_FOLDER = 'static/predections'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDECTION_FOLDER'] = UPLOAD_FOLDER
cors = CORS(app)
app.config['CORS_HEADERS'] = 'application/json'
PIPELINE = '/v1/pipeline'
UI = '/'
DETECTION_URL = "/v1/detect"
DETECTION_BULK_URL = "/v1/"
import os
current_dir = os.getcwd()
# ckp = os.path.join(current_dir, 'derain_dehaze\ckp')
raindrop = os.path.join(current_dir, 'derain_dehaze/ckp/Raindrop.pth')
rainfog = os.path.join(current_dir, 'derain_dehaze/ckp/Rainfog.pth')
save_dir = os.path.join(current_dir, 'static/predections')
original = os.path.join(current_dir, 'images/original')
static_original = os.path.join(current_dir, 'static/original')
static_pred = os.path.join(current_dir, 'static/predections')

import shutil
import cv2





@app.route('/image/<filename>')
def get_image(filename):
    # Check if the file exists
    if True:
        # file_path = f'{os. getcwd()}\images\original\{filename}'
        if os.path.exists(filename):
            return send_file(filename, mimetype='image/jpeg')
        else:
            return f"Error: {filename} not found"
    # else:
    #     file_path = f'{os. getcwd()}\images\predections\{filename}'
    #     if os.path.exists(file_path):
    #         return send_file(filename, mimetype='image/jpeg')
    #     else:
    #         return f"Error: {filename} not found"

@app.route('/videos/<filename>')
def get_video(filename):
    if True:
        # file_path = f'{os. getcwd()}\images\original\{filename}'
        if os.path.exists(filename):
            return send_file(filename, mimetype='video/mp4')
        else:
            return f"Error: {filename} not found"

@app.route(UI)
def mainroute():
    return render_template('index.html')

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/image')
def image():
    return render_template('image.html')


@app.route('/upload', methods = ['GET', 'POST'])
def pipeline():
    '''
    run pipeline on requested video
    '''
    try:
        global raindrop,rainfog,save_dir
        if request.method == 'POST':
            f = request.files['file']
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            new_folder_path = os.path.join(app.config['UPLOAD_FOLDER'], timestamp)
            os.makedirs(new_folder_path, exist_ok=True)

            if f.filename.endswith('.mp4') or f.filename.endswith('.avi') or f.filename.endswith('.MP4') or f.filename.endswith('.AVI'):
                new_predections_folder_path = os.path.join('static/predections', timestamp)
                os.makedirs(new_predections_folder_path, exist_ok=True)

                new_filename = f"{timestamp}_{secure_filename(f.filename)}"
                file_path = os.path.join(new_folder_path, new_filename)
                f.save(file_path)
                file_path = f'{os. getcwd()}/{file_path}'
                
                vidcap = cv2.VideoCapture(file_path)

                success, image = vidcap.read()
                count = 0
                while success:
                    # Save each frame to the folder
                    cv2.imwrite(os.path.join(new_folder_path, f"frame{count}.jpg"), image)
                    success, image = vidcap.read()
                    count += 1
                shutil.move(f'{file_path}', static_original)

                pred_path = pred_detection(raindrop, new_folder_path, new_predections_folder_path)

                 # Compile frames back into a video
                img_array = []
                for i in range(count):
                    img = cv2.imread(os.path.join(new_predections_folder_path, f"frame{i}.jpg"))
                    height, width, layers = img.shape
                    size = (width, height)
                    img_array.append(img)

                out = cv2.VideoWriter(f'{save_dir}/{new_filename}', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

                for i in range(len(img_array)):
                    out.write(img_array[i])
                out.release()

                # Move the output video to the static folder
                # shutil.move('project.avi', static_original)



                
                return render_template('video.html',filename1=f'original/{new_filename}',filename2=f'predections/{new_filename}')

                # return render_template(f'video.html?original={file_path}&pred={file_path}\{new_filename}')
            else:
                new_filename = f"{timestamp}_{secure_filename(f.filename)}"
                file_path = os.path.join(new_folder_path, new_filename)
                f.save(file_path)
                file_path = f'{os. getcwd()}/{file_path}'
                pred_path = pred_detection(raindrop, new_folder_path, save_dir)

                shutil.move(f'{file_path}', static_original)


                
                # shutil.move(pred_path, static_pred)

                # print(pred_path)

                # seg_dir='/home/sumanthreddy/Desktop/Drdo_poc/static/seg/'
                # resp = run(weights='/home/sumanthreddy/Desktop/Drdo_poc/segmentation/yolov7-seg.pt',
                #            name='seg',
                #     source=pred_path,
                #     data = "/home/sumanthreddy/Desktop/Drdo_poc/segmentation/data/coco.yaml",
                #     project = seg_dir,)
                
                # shutil.move(resp, seg_dir)

            
                # res = resp.split('/')[-1]
                return render_template('image.html',filename1=f'original/{new_filename}',filename2=f'predections/{new_filename}',filename3=f'predections/{new_filename}')

                # return render_template('image.html',filename1=f'original/{f.filename}',filename2=f'predections/{f.filename}',filename3=f'seg/{res}')
                # return render_template(f'video.html?original={file_path}&pred={file_path}\{f.filename}')

    except Exception as e:
        print(e)
        return {"success": False, "data": {},"message":str(e)}






@app.route(DETECTION_URL, methods=["POST"])
def predict():
    '''
    Run inference on a single image 
    param: param base
    '''
    try:
        if not request.method == "POST":
            return

        if True:
            request_data = request.json
            now = datetime.datetime.now()
            dt_string = now.strftime("%d-%m-%H-%M-%S")
            # filename = str(random.randint(1, 80000))+'.jpg'
            filename = dt_string+'.jpg'
            # print(data['base'])
            with open(image_files+filename, "wb") as fh:
                fh.write(base64.urlsafe_b64decode(request_data['base']))
            img = Image.open(image_files+filename)
            results = model(img)

            crops = results.crop(save=False)
            # data = []
            data = {
                "total_count": len(crops),
                "head_count": 0,
                "shrimp_count": 0,
            }
            for crop in crops:
                # print(/crop)
                file_name = image_files+"detections/" + \
                    classes[str(int(crop['cls'].item()))]['actual'] + \
                    str(random.randint(1, 80000))+".jpg"
                cv2.imwrite(file_name, cv2.cvtColor(
                    crop['im'], cv2.COLOR_RGB2BGR))
                # img = Image.open(file_name)
                if str(int(crop['cls'].item())) == 0:
                    data['head_count'] += 1
                else:
                    data['shrimp_count'] += 1



            # results.imgs
            results.render()
            for im in results.imgs:
                buffered = BytesIO()
                im_base64 = Image.fromarray(im)
                im_base64.save(buffered, format="JPEG")
                final_image = base64.b64encode(
                    buffered.getvalue()).decode('utf-8')

            return {"success": True, "data": data, "image": final_image,"message": "Image processed successfully"}

    except Exception as e:
        print(e)
        return {"success": False, "data": {},"message":str(e)}





@app.route(DETECTION_BULK_URL, methods=["GET"])
def bulk_predict():
    '''
    Run inference on a bulk images image
    '''
    try:
        for image_path in glob("image_files/*.jpg"):
            img = Image.open(image_path)
            results = model(img)
            results.crop(save=True)
            results.render()
            for im in results.imgs:
                buffered = BytesIO()
                im_base64 = Image.fromarray(im)
                im_base64.save(buffered, format="JPEG")
                final_image = base64.b64encode(
                    buffered.getvalue()).decode('utf-8')
                now = datetime.datetime.now()
                dt_string = now.strftime("%d-%m-%H-%M-%S")
                filename = dt_string+'.jpg'
                with open(ocr_files+'imprint/'+filename, "wb") as fh:
                    fh.write(base64.urlsafe_b64decode(final_image))

    except Exception as e:
        print(e)
        return {"success": False, "data": "data"}





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)

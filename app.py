import os, sys
from flask import Flask, request, jsonify, render_template, Response 
from flask_cors import CORS, cross_origin 

from src.DermaCareNet.exception import ComputerVisionYolov11Exception
from src.DermaCareNet.groq_agents.llm_agent_with_tools import skin_disease_agent
from src.DermaCareNet.utils.common_utils import decodeImage, image_encode_base64


from ultralytics import YOLO

import requests
import shutil
from typing import Tuple, List

image_path : str = r"C:\Users\VISHNU\Desktop\DermaCareNet\DermaCareNet-AI\data\inputImage.jpg"
best_model: str = r"C:\Users\VISHNU\Desktop\DermaCareNet\DermaCareNet-AI\model\best\best.pt"
predict_path: str = r"C:\Users\VISHNU\Desktop\DermaCareNet\DermaCareNet-AI\runs\detect\predict\inputImage.jpg"


app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"




@app.route("/")
def home():
    return render_template("index.html")



def detect_and_predict_class(img_path: str, best_model: str) -> Tuple[List[str], List[float]]:
    try:
        model = YOLO(best_model)
        results = model(img_path, save=True) 
        names = model.names  
        boxes = results[0].boxes 

        cls_names = []
        confs = []

        print("Detected Classes:")
        for box in boxes:
            cls_id = int(box.cls)  # Class ID
            cls_name = names[cls_id]  # Class name
            conf = float(box.conf)

            print(f"{cls_name}: {conf:.2f}")
            cls_names.append(cls_name)
            confs.append(conf)

        return cls_names, confs

    except Exception as e:
        raise ComputerVisionYolov11Exception(e, sys)


@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predictRoute() -> Response:

    try:
        image = request.json["image"]
        decodeImage(image, clApp.filename)

        clss_names, conf = detect_and_predict_class(
            img_path=image_path,
            best_model=best_model

        )

        open_encode_base64 = image_encode_base64(
            image_path=predict_path
        )
        
        encoded_image = open_encode_base64.decode('utf-8')
        
        folder_path = "runs"        

        if os.path.exists(folder_path):
            shutil.rmtree(folder_path, ignore_errors=True)

        if clss_names:
            top_classes = clss_names 
            duplicte_remove = set(top_classes)
            primary_condition = ", ".join(duplicte_remove) 
        else:
            primary_condition = "No specific condition detected"    

        agent_executor = skin_disease_agent(
            model="llama-3.3-70b-versatile"
        )    

        agent_input = {
            "detected_conditions": primary_condition 
            
        }

        try:
            result = agent_executor.invoke(agent_input)
            recommendations = result.get("output", "Recommendations were not found in the agent's response.")
        except Exception as agent_error:
            print(f"Error during agent execution: {agent_error}")
            recommendations = f"An error occurred while generating recommendations: {str(agent_error)}" 

        response = {
            "result_image": encoded_image,
            "detected_conditions": primary_condition,
            "confidence": conf,
            "recommendations": recommendations,
        
        }      

        return jsonify(response)    

    except Exception as e:
        raise ComputerVisionYolov11Exception(e, sys)    








if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host="0.0.0.0", port=8080, debug=True)
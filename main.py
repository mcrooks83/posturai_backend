# postrai_backend/main.py
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
import uvicorn
import cv2
import numpy as np
import mediapipe as mp
import io
from starlette.datastructures import UploadFile
from analyse_posture import analyze_posture_and_annotate 
import os
import httpx
import base64
from dotenv import load_dotenv

load_dotenv()  # loads variables from .env
# openAi key


# Pose setup
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True)

# Helper to extract XY

def extract_landmarks(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)

    if not results.pose_landmarks:
        return None, None, None

    landmarks = results.pose_landmarks.landmark
    keypoints = {
        name: (int(lm.x * w), int(lm.y * h))
        for name, lm in enumerate(landmarks)
    }
    return keypoints, image.shape, results.pose_landmarks

def analyze_posture(keypoints):
    # For now, return the raw keypoints
    return {k: {'x': v[0], 'y': v[1]} for k, v in keypoints.items()}

async def analyze(request: Request):
    form = await request.form()
    upload: UploadFile = form["file"]
    contents = await upload.read()

    keypoints, shape, landmarks = extract_landmarks(contents)
    if keypoints is None:
        return JSONResponse({"error": "No pose detected"}, status_code=400)

    # Decode the image again to annotate
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Draw landmarks on image using mediapipe drawing utils
    #mp.solutions.drawing_utils.draw_landmarks(
    #    image, landmarks, mp_pose.POSE_CONNECTIONS)

     # Call your posture analysis + annotation function
    annotated_image, report_text = analyze_posture_and_annotate(image.copy(), landmarks)

    # Encode annotated image to PNG in memory
    #success, buffer = cv2.imencode('.png', image)
    success, buffer = cv2.imencode('.png', annotated_image)
    if not success:
        return JSONResponse({"error": "Failed to encode image"}, status_code=500)

    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse({
        "posture": report_text,
        "image_base64": img_base64
    })

async def get_exercise_recommendation(request: Request):
    try:
        data = await request.json()
        posture_report = data.get("posture_report")
        #print(posture_report, flush=True)
        if not posture_report:
            return JSONResponse({"error": "Missing posture_report in JSON body"}, status_code=400)

        prompt = (
            "You are a helpful physical therapist. "
            "Provide simple English exercises based on this posture report:\n\n"
            f"{posture_report}"
        )

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        json_body = {
           "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful physical therapist."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.7,
        }

        if not OPENAI_API_KEY:
            raise ValueError("Missing OpenAI API key")


        async with httpx.AsyncClient() as client:
            
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=json_body,
                timeout=30
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                error_body = await exc.response.aread()
                print("API returned error:", error_body.decode(), flush=True)
                return JSONResponse({"error": "OpenAI API error", "details": error_body.decode()}, status_code=exc.response.status_code)
            data = await response.json()
            print(data, flush=True)
            choices = data.get("choices", [])
            if choices:
                recommendation = choices[0]["message"]["content"].strip()
            else:
                recommendation = "No recommendation available."

        return JSONResponse({"exercise_recommendation": recommendation})

    except Exception as e:
        return JSONResponse({"error": "Failed to get exercise recommendation", "details": str(e)}, status_code=500)


routes = [
    Route("/analyze", analyze, methods=["POST"]),
    Route("/exercise", get_exercise_recommendation, methods=["POST"]),
]

app = Starlette(debug=True, routes=routes)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

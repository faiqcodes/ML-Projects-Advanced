from fastapi import FastAPI, UploadFile, File
import torch
import cv2
import numpy as np
from model import UNet
import os

app = FastAPI(title="Brain MRI Tumor Detection")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model initialize kar
model = UNet(in_channels=3, out_channels=1).to(device)
model.eval()

model_path = r"E:\ML - GITHUB\ML-Projects-Advanced\04_Brain_Mri_model\Models\Unet_Model_weights.pth"

# Verify weights load ho rahe hain
try:
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        print(f"File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check karo checkpoint kya format hai
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {checkpoint.keys()}")
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        # Verify - pehli layer ke weights check kar
        first_layer_weights = list(model.parameters())[0]
        print(f"✅ Model weights loaded successfully!")
        print(f"First layer weight sample: {first_layer_weights[0, 0, :3, :3]}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"status": "error", "message": "Image load nahi hua"}
        
        img = cv2.resize(img, (256, 256))
        img_normalized = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).to(device)
        img_tensor = img_tensor.float()
        
        # Model prediction
        with torch.no_grad():
            output = model(img_tensor)
            output = torch.sigmoid(output)
        
        tumor_mask = output.squeeze().cpu().numpy()
        tumor_mask = np.clip(tumor_mask, 0, 1)
        
        
        binary_mask = (tumor_mask > 0.5).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        tumor_locations = np.where(binary_mask == 1)
        
        if len(tumor_locations[0]) > 0:
            y_min, y_max = int(tumor_locations[0].min()), int(tumor_locations[0].max())
            x_min, x_max = int(tumor_locations[1].min()), int(tumor_locations[1].max())
            tumor_pixels = len(tumor_locations[0])
            tumor_percentage = (tumor_pixels / (256*256)) * 100
            y_center = int(tumor_locations[0].mean())
            x_center = int(tumor_locations[1].mean())
        else:
            x_min = x_max = y_min = y_max = 0
            x_center = y_center = 128
            tumor_percentage = 0
            tumor_pixels = 0
        
        is_tumor_detected = tumor_percentage > 0.5
        
        return {
            "status": "success",
            "tumor_detected": is_tumor_detected,
            "tumor_percentage": round(tumor_percentage, 2),
            "tumor_location": {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "center_x": x_center,
                "center_y": y_center,
                "tumor_area_pixels": tumor_pixels
            }
        }
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/")
def home():
    return {"message": "Brain MRI Tumor Detection API", "status": "running"}
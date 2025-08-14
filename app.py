import streamlit as st
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os

# Define the device for model inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Architecture Classes (from training.docx) ---
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAMBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg = self.mlp(self.avg_pool(x))
        max_val = self.mlp(self.max_pool(x))
        channel_att = self.sigmoid_channel(avg + max_val)
        x = x * channel_att
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial(torch.cat([avg_out, max_out], dim=1))
        return x * spatial_att

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv6 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.conv12 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.conv18 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
        self.out_conv = nn.Conv2d(out_channels * 4, out_channels, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv6(x)
        x3 = self.conv12(x)
        x4 = self.conv18(x)
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        output = self.out_conv(x_cat)
        return output

class EnhancedUNet(nn.Module):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', classes=1):
        super(EnhancedUNet, self).__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=1, 
            classes=classes,
            activation=None
        )
        # Use the correct sample input size from your training script
        sample_input = torch.randn(1, 1, 128, 128) 
        with torch.no_grad():
            encoder_features = self.unet.encoder(sample_input)
        
        # Get the number of channels for the ASPP/CBAM input
        aspp_in_channels = encoder_features[-2].shape[1]
        self.cbam = CBAMBlock(aspp_in_channels)
        self.aspp = ASPP(aspp_in_channels, aspp_in_channels)

    def forward(self, x):
        features = self.unet.encoder(x)
        feature_for_aspp = features[-2]
        context = self.aspp(feature_for_aspp)
        attn = self.cbam(context)
        modified_features_list = list(features)
        modified_features_list[-2] = attn
        # Pass the list directly to the decoder
        decoder_out = self.unet.decoder(modified_features_list)
        seg = self.unet.segmentation_head(decoder_out)
        return seg

# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    model = EnhancedUNet().to(device)
    state_dict = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Define the preprocessing transformation
preprocess_transform = A.Compose([
    A.Normalize(mean=(0.5,), std=(0.25,)),
    ToTensorV2()
])

def predict(model, image):
    # Ensure the image is in grayscale (L) and convert to a numpy array
    image_np = np.array(image.convert("L"))
    
    # Apply the same preprocessing as in the training script
    augmented = preprocess_transform(image=image_np)
    input_tensor = augmented['image'].unsqueeze(0).to(device)
    
    # Get the prediction from the model
    with torch.no_grad():
        output = model(input_tensor)
    
    # Apply sigmoid to get probabilities and create a binary mask
    sigmoid_output = torch.sigmoid(output).squeeze().cpu().numpy()
    pred_mask = (sigmoid_output > 0.5).astype(np.uint8) * 255
    
    return pred_mask

# --- Streamlit UI ---
st.title("Brain Tumor Segmentation App")
st.write("Upload an MRI scan to get the predicted tumor segmentation mask.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an MRI image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Load the model
    model = load_model()

    # Run prediction
    with st.spinner("Processing image..."):
        predicted_mask = predict(model, image)
    
    # Display the predicted mask
    st.image(predicted_mask, caption="Predicted Tumor Mask", use_column_width=True)
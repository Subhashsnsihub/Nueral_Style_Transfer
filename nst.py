import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, x):
        gram = gram_matrix(x)
        self.loss = F.mse_loss(gram, self.target)
        return x

def gram_matrix(input):
    batch_size, f_map_num, h, w = input.size()
    features = input.view(batch_size * f_map_num, h * w)
    G = torch.mm(features, features.t())
    return G.div(batch_size * f_map_num * h * w)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

def image_loader(image_path, size=512):
    try:
        image = Image.open(image_path).convert('RGB')
        loader = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor()])
        image = loader(image).unsqueeze(0)
        return image
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                              style_img, content_img):
    model = nn.Sequential()
    normalization = Normalization(normalization_mean, normalization_std)
    model.add_module("normalization", normalization)
    
    content_losses = []
    style_losses = []
    
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue

        model.add_module(name, layer)

        if name in {'conv_4'}:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in {'conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'}:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    return model, style_losses, content_losses

def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=300,
                      style_weight=1000000, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn,
        torch.tensor([0.485, 0.456, 0.406]),
        torch.tensor([0.229, 0.224, 0.225]),
        style_img, content_img)

    optimizer = torch.optim.LBFGS([input_img.requires_grad_()])
    
    progress_bar = st.progress(0)
    step = [0]
    
    while step[0] < num_steps:  # Changed <= to < to avoid going over 1.0
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            
            style_score = 0
            content_score = 0
            
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
                
            style_score *= style_weight
            content_score *= content_weight
            
            loss = style_score + content_score
            loss.backward()
            
            step[0] += 1
            # Ensure progress value stays between 0 and 1
            progress = min(step[0] / num_steps, 1.0)
            progress_bar.progress(progress)
            
            return style_score + content_score
        
        optimizer.step(closure)
    
    input_img.data.clamp_(0, 1)
    return input_img

def main():
    st.set_page_config(page_title="Neural Style Transfer", layout="wide")
    
    st.title("Neural Style Transfer")
    st.write("Upload your content and style images to create an artistic fusion!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Settings")
        image_size = st.selectbox(
            "Select image size (larger = better quality but slower)",
            options=[128, 224, 512],
            index=1
        )
        
        style_weight = st.slider(
            "Style Weight (higher = more stylistic)",
            min_value=1e4,
            max_value=1e7,
            value=1e6,
            format="%e"
        )
        
        num_steps = st.slider(
            "Number of Steps (higher = better quality but slower)",
            min_value=100,
            max_value=500,
            value=300
        )
    
    with col2:
        st.subheader("Upload Images")
        content_file = st.file_uploader("Choose Content Image", type=['png', 'jpg', 'jpeg'])
        style_file = st.file_uploader("Choose Style Image", type=['png', 'jpg', 'jpeg'])

    if content_file and style_file:
        try:
            with st.spinner("Loading images..."):
                content_img = image_loader(content_file, size=image_size)
                style_img = image_loader(style_file, size=image_size)
                
                if content_img is None or style_img is None:
                    st.error("Error loading images. Please try different images.")
                    return
                    
                input_img = content_img.clone()

            col1, col2 = st.columns(2)
            with col1:
                st.write("Content Image:")
                st.image(content_file, width=300)
            with col2:
                st.write("Style Image:")
                st.image(style_file, width=300)

            if st.button("Start Style Transfer"):
                st.warning("Running on CPU mode. This process might take several minutes.")
                
                with st.spinner("Loading VGG19 model..."):
                    cnn = models.vgg19(pretrained=True).features.eval()

                try:
                    st.write("Processing... Please wait...")
                    output = run_style_transfer(
                        cnn, 
                        content_img, 
                        style_img, 
                        input_img,
                        num_steps=num_steps,
                        style_weight=style_weight
                    )
                    
                    output_img = output.cpu().squeeze(0)
                    output_img = transforms.ToPILImage()(output_img)
                    
                    st.success("Style transfer completed!")
                    st.write("Generated Image:")
                    st.image(output_img, width=300)
                    
                    buf = io.BytesIO()
                    output_img.save(buf, format='PNG')
                    st.download_button(
                        label="Download Generated Image",
                        data=buf.getvalue(),
                        file_name="styled_image.png",
                        mime="image/png"
                    )
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        st.error("Out of memory error. Please try a smaller image size.")
                    else:
                        st.error(f"An error occurred during processing: {str(e)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please try again with different images or check your system requirements.")

if __name__ == "__main__":
    main()
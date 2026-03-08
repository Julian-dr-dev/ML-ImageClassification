import torch 
from torchvision import transforms
from PIL import Image
import sys
from model import Net

device = "cuda" if torch.cuda.is_available() else "cpu"

clases = ["cat", "dog", "wild"]

model = Net(len(clases))
model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# Image path from terminal
image_path = sys.argv[1]

image = Image.open(image_path). convert("RGB")
image = transform(image).unsqueeze(0).to(device)


with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

prediction = classes[predicted.item()]

print("Prediction:", prediction)

#Libraries

import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.autograd import Variable
from PIL import Image
import pathlib
import glob

# Paths

train_path = '//Path to TRAINING DATA// I used the IntelImageClassification set that was available on Internet'
pred_path = '//Path to TRAINING DATA// I used the IntelImageClassification set that was available on Internet'

# Categories

root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])

#ConvNet Class
class ConvNet(nn.Module):
    def __init__(self, num_classes=6):
        super(ConvNet, self).__init__()

        # Output size after convolution filter
        # ((w-f+2P)/s) +1 to find Height and width of O/P
        # In first case Width w=150, Kernel Size f=3, padding =1 and Stride = 1

        # Input shape= (256,3,150,150)
        #      where 256 Batch Size, 3 num of channels(RGB), 150x150 is HxW


        # Create CNN, Then Batch Normalize and then use ReLU to bring the Non-Linearity.

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        # Shape= (256,12,150,150)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # Shape= (256,12,150,150)
        self.relu1 = nn.ReLU()
        # Shape= (256,12,150,150)

        # To reduce the size of the op cn layer by 2;
        self.pool = nn.MaxPool2d(kernel_size=2)
        # Reduce the image size be factor 2
        # Shape= (256,12,75,75)

        # 2nd CN Layer
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        # Shape= (256,20,75,75)
        self.relu2 = nn.ReLU()
        # Shape= (256,20,75,75)

        # 3rd CN Layer
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Shape= (256,32,75,75)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        # Shape= (256,32,75,75)
        self.relu3 = nn.ReLU()
        # Shape= (256,32,75,75)

        # Last Layer
        self.features_conv = self.conv3
        # placeholder for the gradients
        self.gradients = None


        # Add fully connected Layer
        self.fc = nn.Linear(in_features=75 * 75 * 32, out_features=num_classes)

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.3)

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)
        # register the hook
        h = x.register_hook(self.activations_hook)

        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

        # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
   def get_activations(self, x):
       return self.features_conv(x)

    # Feed forward function

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        # Apply dropout
        output = self.dropout(output)

        # Above output will be in matrix form, with shape (256,32,75,75)
        # We should reshape matrix to feed it to the Fully Connected Layer
        output = output.view(-1, 32 * 75 * 75)

        output = self.fc(output)

        return output


#Checkpoint Load

checkpoint=torch.load('best_checkpoint.model')
model=ConvNet(num_classes=6)
model.load_state_dict(checkpoint) #Feed checkpoint to the class
model.eval()   #To set dropout and normalization layers to Eval Mode

#Transforms
#Note: Here we dont need Flip or such operations
transformer=transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,0.5,0.5])
])



#Prediction function
def prediction(img_path, transformer):

    image = Image.open(img_path)
    #Feed it to transformer to convert it to tensor
    image_tensor = transformer(image).float()
    #Extra batch dimension since PyTorch treats everything as batches
    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor.cuda()

    #Convert to variable
    input = Variable(image_tensor)

    #Make prediction using model with input
    output = model(input)
    #To get cat id with max probability
    index = output.data.numpy().argmax()
    #Get category name
    pred = classes[index]

    return pred

#Fetch all image path in seg_pred folder and save it inside images_path
images_path=glob.glob(pred_path+'/*.jpg')
#Create Empty dic
pred_dict={}

#Save Image name as key and Pred as value.
for i in images_path:
    pred_dict[i[i.rfind('/')+1:]]=prediction(i,transformer)

print(pred_dict)
from torchvision import transforms
from torchvision import models
from PIL import Image
import torch
import os

#set wd
path = '/home/jun/Documents/Programming/DL_With_PyTorch/Book_git/dlwpt-code/data/p1ch2/' 
os.chdir(path)
print(os.getcwd())


#get image
img = Image.open('bobby.jpg')
#print(img)
img.show()
print('got here')

#preprocess fn
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )])
#process bobby image
img_t = preprocess(img)
#unsqueeze
batch_t = torch.unsqueeze(img_t, 0)

print('CREATING MODEL \n\n\n\n\n')

resnet = models.resnet101(pretrained = True)
#put network in eval mode, for inference: running trained model on new data
resnet.eval()

out = resnet(batch_t)
#print(out)


print('IMPORTING LABELS  \n\n\n\n\n')

#load imagenet labels
with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]
#    print(labels)


#find index of max score in previous tensor, returns 1-elem., 1-dim. tensor: out([207])
_, index = torch.max(out, 1)

#convert top score to prob. and match to text label
percentage = torch.nn.functional.softmax(out, dim = 1)[0] * 100
match = labels[index[0]]
print(match)
print(percentage[index[0]].item())

#sort scores, create list
_, indices = torch.sort(out, descending = True) #indices is tensor of n elems.
top_scores = [(labels[idx], percentage[idx].item()) for idx in indices[0][:10]] 
print(top_scores)


from torchvision import models

print(dir(models))

print('Hello World')

resnet = models.resnet101(pretrained = True)

print(resnet)



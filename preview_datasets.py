import os
from matplotlib import pyplot as plt

subset = "rock"
cwd = os.getcwd()
train_dir = os.path.join(cwd, "datasets", "train")
test_dir = os.path.join(cwd, "datasets", "test")
val_dir = os.path.join(cwd, "datasets", "val")

img_path = os.path.join(train_dir, subset)
img_name = os.listdir(img_path)

for i in range(9):
    plt.subplot(3, 3, i+1)
    # plt.xticks([])
    # plt.yticks([])
    plt.imshow(plt.imread(os.path.join(img_path, img_name[i])))
    plt.xlabel(img_name[i])
plt.show()

print(img_name[:4])
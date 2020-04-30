path = '__Path to cell segmentation gold truth __ /01_GT/SEG'
names = os.listdir(path)
names = [name for name in names if '.tif' in name and 'seg' in name]
names.sort()
print(names)
path2 = '__Path to network segmentation results__ /01_RES'
names2 = []
for i, name in enumerate(names):
    name = names[i]
    names2.append(name.replace('man_seg', 'mask'))
names2.sort()
print(names2)

ic_scores = np.zeros(len(names))

for i, name in enumerate(names):
    gt_img = io.imread(os.path.join(path, name))
    gt_img = skimage.img_as_float(gt_img)
    print(gt_img.shape)
    print('image dtype ', gt_img.dtype)
    plt.imshow(gt_img)
    plt.title('gt_img')
    plt.show()
    pred_img = io.imread(os.path.join(path2, names2[i]))
    pred_img = skimage.img_as_float(pred_img)
    print(pred_img.shape)
    plt.imshow(pred_img)
    plt.title('pred_img')
    plt.show()

avg_ic = np.mean(ic_scores)
std_ic = np.std(ic_scores)

print("Scores: ")
print(ic_scores)

print("Average: ")
print(avg_ic)
print("Standard Deviation: ")
print(std_ic)
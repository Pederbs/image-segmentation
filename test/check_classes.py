import matplotlib.pyplot as plt

img_path = '/home/peder/image-segmentation/skagerakk_dataset/annotations_prepped_train/scaled/2022-08-22.png'


# Load the image
img = plt.imread(img_path)
print(img.shape)
# img = img[:, :, :1]  # Ensure it's grayscale
# show image histogram
# img = img*255
# img = img.astype('uint8')  # Ensure the image is in uint8 format

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.hist(img.ravel(), bins=5, color='blue', alpha=0.7)
print(f"Image shape: {img.shape}")
print(f"Unique pixel values: {set(img.ravel())}")
plt.title('Histogram')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')
plt.xlim([-0.0001, 0.008])
plt.savefig('image_histogram.png', bbox_inches='tight')


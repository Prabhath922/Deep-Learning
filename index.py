from fastai.vision.all import *
from fastdownload import download_url
import time

# Let's use a different approach since DuckDuckGo is rate limiting us
# We'll use a pre-existing dataset or manually upload images

# Option 1: Use a pre-existing dataset (recommended)
path = untar_data(URLs.PETS)/'images'

# If you want to specifically work with cats vs dogs, let's filter the dataset
def is_cat(x): return x[0].isupper()

dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

# Let's take a look at our data
dls.show_batch(max_n=6)

# Create a vision_learner and fine-tune it
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3) # Train for 3 epochs

# Test with a sample image
# Upload an image or use one from the dataset
img = PILImage.create(get_image_files(path)[0])
is_cat, _, probs = learn.predict(img)
print(f"Is this a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1]:.4f}")
# Image Colorization
 
# <font color=white><center><b>**ColorGAN – Image Colorization using Generative Adversarial Networks**</center></b></font>

## Project Overview
This project focuses on implementing an image colorization model using Generative Adversarial Networks (GANs). The goal is to automatically transform grayscale images into realistic colored images, leveraging the adversarial training between a **Generator** and a **Discriminator**.

## Dependencies
The project requires the following libraries:
- `numpy`
- `pandas`
- `opencv-python`
- `matplotlib`
- `tensorfls`
- `tqdm`
- `scikit-learn`
- `seaborn`

## Key Steps in the Project

1. **Data Preparation**:
   - Grayscale and color image datasets are loaded, resized to **256x256 pixels**, normalized, and converted into NumPy arrays.  
   - Images are then organized into TensorFlow datasets for training and testing.

2. **Model Architecture**:
   - **Generator**: A U-Net-based architecture with downsampling and upsampling layers, skip connections, and a final `tanh` activation to generate colored images from grayscale input.  
   - **Discriminator**: A PatchGAN-based classifier that distinguishes between real colored images and generated ones by evaluating local image patches.

3. **Loss Functions**:
   - **Generator Loss**: Combines adversarial loss with an L1 loss (mean absolute error) to encourage realistic and perceptually accurate colorization.  
   - **Discriminator Loss**: Uses binary cross-entropy to classify real vs. generated image pairs.

4. **Training Process**:
   - The model is trained adversarially for 100 epochs.  
   - At each step, the generator creates a colorized image, while the discriminator evaluates its realism against ground truth color images.  
   - Optimizers: **Adam** with learning rate `2e-4` and `beta1=0.5`.

5. **Visualization**:
   - Sample predictions are displayed alongside their grayscale input and ground-truth color images.  
   - Training loss curves for both the Generator and Discriminator are plotted for performance monitoring.

6. **Evaluation**:
   - The model’s effectiveness is visually assessed through side-by-side comparison of generated vs. ground-truth color images.  
   - Loss plots demonstrate the convergence behavior of GAN training.

## Results
- The trained GAN successfully colorizes grayscale images, produg between the Generator and Discriminator.  
- Visual comparisons indicate that the generator learns to restore colors ths` function to visualize predictions on test samples.

## Conclusion
This project demonstrates the power of GANs in the field of computer vision, specifically for image-to-image translation tasks such as **automatic colorization**. The results highlight how adversarial learning combined with perceptual loss functions can generate visually convincing and context-aware colorized images.

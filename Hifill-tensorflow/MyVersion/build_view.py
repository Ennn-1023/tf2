import argparse
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import tensorflow as tf
import csv

def calculate_psnr_ssim(img1, img2):
    img1 = tf.convert_to_tensor(img1, dtype=tf.float32)
    img2 = tf.convert_to_tensor(img2, dtype=tf.float32)
    psnr_value = tf.image.psnr(img1, img2, max_val=255.0)
    ssim_value = tf.image.ssim(img1, img2, max_val=255.0)
    return psnr_value.numpy(), ssim_value.numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--raw', type=str, help='Path to the raw image')
    parser.add_argument('-w', '--white', type=str, help='Path to the white result')
    parser.add_argument('-i', '--inpainted', type=str, help='Path to the inpainted result')
    parser.add_argument('-g', '--gan', type=str, default=None, help='Path to the gan result')
    parser.add_argument('-o', '--output', type=str, help='Path to the output dir')

    opt, _ = parser.parse_known_args()
    
    whites = os.listdir(opt.white)
    whites.sort()

    # Collect results for CSV
    results = []

    with tqdm(total=len(whites), desc="generating views", unit="image") as pbar:
        for white in whites:
            fname = white.replace('_raw.jpg', '')
            raw = plt.imread(os.path.join(opt.raw, f'{fname}_raw.jpg'))
            raw = cv2.resize(raw, (512, 512))
            white_img = plt.imread(os.path.join(opt.white, f'{fname}_raw.jpg'))
            white_img = cv2.resize(white_img, (512, 512))
            fixed = plt.imread(os.path.join(opt.inpainted, f'{fname}_raw.jpg'))
            fixed = cv2.resize(fixed, (512, 512))
            if opt.gan is not None:
                gan = plt.imread(os.path.join(opt.gan, f'{fname}_raw.jpg'))
                gan = cv2.resize(gan, (512, 512))
                grid = 5
            else:
                gan = None
                grid = 4

            input_img = plt.imread(os.path.join(opt.raw, f'{fname}_fixed.jpg'))
            input_img = cv2.resize(input_img, (512, 512))
                        
            # Plotting
            ax = plt.subplot(1, grid, 1)
            ax.imshow(raw)
            ax.axis('off')
            ax.set_title('GT')

            ax = plt.subplot(1, grid, 2)
            ax.imshow(input_img)
            ax.axis('off')
            ax.set_title('Input')

            ax = plt.subplot(1, grid, 3)
            ax.imshow(white_img)
            ax.set_title('White')
            ax.axis('off')
            white_psnr, white_ssim = calculate_psnr_ssim(raw, white_img)
            ax.text(0, 600, f'PSNR:{white_psnr:.2f}, SSIM:{white_ssim:.2f}', fontsize=4, ha='left', va='bottom')

            ax = plt.subplot(1, grid, 4)
            ax.imshow(fixed)
            ax.set_title('Inpainted')
            ax.axis('off')
            inpainted_psnr, inpainted_ssim = calculate_psnr_ssim(raw, fixed)
            ax.text(0, 600, f'PSNR:{inpainted_psnr:.2f}, SSIM:{inpainted_ssim:.2f}', fontsize=4, ha='left', va='bottom')

            if gan is not None:
                ax = plt.subplot(1, 5, 5)
                ax.imshow(gan)
                ax.set_title('GAN')
                ax.axis('off')
                gan_psnr, gan_ssim = calculate_psnr_ssim(raw, gan)
                ax.text(0, 600, f'PSNR:{gan_psnr:.2f}, SSIM:{gan_ssim:.2f}', fontsize=4, ha='left', va='bottom')
            else:
                gan_psnr, gan_ssim = None, None
            
            # Save image
            if not os.path.exists(opt.output):
                os.makedirs(opt.output, exist_ok=True)
            plt.savefig(os.path.join(opt.output, f'{fname}.jpg'), dpi=500, facecolor='white', bbox_inches='tight', pad_inches=0)
            plt.clf()

            # Save PSNR/SSIM to results
            result_row = {
                "image": f'{fname}_raw.jpg',
                "white_psnr": white_psnr,
                "white_ssim": white_ssim,
                "inpainted_psnr": inpainted_psnr,
                "inpainted_ssim": inpainted_ssim
            }
            if gan is not None:
                result_row["gan_psnr"] = gan_psnr
                result_row["gan_ssim"] = gan_ssim
            results.append(result_row)

            pbar.update(1)

    # # Write results to CSV
    # csv_path = os.path.join(opt.output, "results.csv")
    # with open(csv_path, 'w', newline='') as csvfile:
    #     fieldnames = ["image", "white_psnr", "white_ssim", "inpainted_psnr", "inpainted_ssim"]
    #     if opt.gan is not None:
    #         fieldnames += ["gan_psnr", "gan_ssim"]
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     writer.writerows(results)

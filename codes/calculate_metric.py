import os
import math
import numpy as np
import cv2
import glob
from sewar.full_ref import sam, uqi, scc

def main():
    # Configurations (defaults match the SR images written by test_ucm_x3.sh)
    folder_GT = './datasets/UCMerced-dataset/test/HR_x3'
    folder_Gen = '../experiment/AERUNet_UCMerced_x3/results'

    # Write the result file next to the folder of generated images
    result_file = os.path.join(os.path.dirname(folder_Gen), 'metrics.txt')

    crop_border = 3  # same as scale
    suffix = '_x3_SR'  # suffix of the SR images saved by --save_results
    test_Y = False

    # [Rest of the code remains exactly the same...]
    PSNR_all = []
    SSIM_all = []
    SAM_all = []
    QI_all = []
    SCC_all = []

    img_list = sorted(glob.glob(folder_GT + '/*'))

    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    # Open file for writing results
    with open(result_file, 'w') as f:
        if test_Y:
            print('Testing Y channel.')
            f.write('Testing Y channel.\n')
        else:
            print('Testing RGB channels.')
            f.write('Testing RGB channels.\n')

        f.write('\nPer-image results:\n')
        f.write('-' * 100 + '\n')
        f.write('{:<4} {:<25} {:>10} {:>10} {:>10} {:>10} {:>10}\n'.format(
            'No.', 'Image Name', 'PSNR', 'SSIM', 'SAM', 'QI', 'SCC'))
        f.write('-' * 100 + '\n')

        for i, img_path in enumerate(img_list):
            try:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                im_GT = cv2.imread(img_path) / 255.
                im_Gen = cv2.imread(os.path.join(folder_Gen, base_name + suffix + '.png')) / 255.

                if test_Y and im_GT.shape[2] == 3:
                    im_GT_in = bgr2ycbcr(im_GT)
                    im_Gen_in = bgr2ycbcr(im_Gen)
                else:
                    im_GT_in = im_GT
                    im_Gen_in = im_Gen

                if crop_border == 0:
                    cropped_GT = im_GT_in
                    cropped_Gen = im_Gen_in
                else:
                    if im_GT_in.ndim == 3:
                        cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border, :]
                        cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border, :]
                    elif im_GT_in.ndim == 2:
                        cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border]
                        cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border]
                    else:
                        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT_in.ndim))

                # Calculate metrics with error handling
                try:
                    PSNR = calculate_rgb_psnr(cropped_GT * 255, cropped_Gen * 255)
                except:
                    PSNR = 0
                    print(f"Error calculating PSNR for {base_name}")

                try:
                    SSIM = calculate_ssim(cropped_GT * 255, cropped_Gen * 255)
                except:
                    SSIM = 0
                    print(f"Error calculating SSIM for {base_name}")

                try:
                    SAM_val = sam(cropped_GT * 255, cropped_Gen * 255)
                    SAM = SAM_val if SAM_val is not None else 0
                except:
                    SAM = 0
                    print(f"Error calculating SAM for {base_name}")

                try:
                    QI_val = uqi(cropped_GT * 255, cropped_Gen * 255)
                    QI = QI_val if QI_val is not None else 0
                except:
                    QI = 0
                    print(f"Error calculating QI for {base_name}")

                try:
                    SCC_val = scc(cropped_GT * 255, cropped_Gen * 255)
                    SCC = SCC_val if SCC_val is not None else 0
                except:
                    SCC = 0
                    print(f"Error calculating SCC for {base_name}")

                # Print to console
                print('{:3d} - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}, \tSAM: {:.6f}, \tQI: {:.6f}, \tSCC: {:.6f}'.format(
                    i + 1, base_name, PSNR, SSIM, SAM, QI, SCC))
                
                # Write to file
                f.write('{:<4d} {:<25} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f}\n'.format(
                    i + 1, base_name, PSNR, SSIM, SAM, QI, SCC))

                # Only append valid metrics
                if PSNR > 0: PSNR_all.append(PSNR)
                if SSIM > 0: SSIM_all.append(SSIM)
                if SAM > 0: SAM_all.append(SAM)
                if QI > 0: QI_all.append(QI)
                if SCC > 0: SCC_all.append(SCC)

            except Exception as e:
                print(f"Error processing image {base_name}: {str(e)}")
                continue

        # Calculate and write averages (only if we have valid metrics)
        f.write('\n' + '-' * 100 + '\n')
        f.write('Average Metrics:\n')
        f.write('-' * 100 + '\n')

        def safe_average(metric_list):
            return sum(metric_list) / len(metric_list) if metric_list else 0

        avg_psnr = safe_average(PSNR_all)
        avg_ssim = safe_average(SSIM_all)
        avg_sam = safe_average(SAM_all)
        avg_qi = safe_average(QI_all)
        avg_scc = safe_average(SCC_all)

        # Print averages to console
        print('\nAverage: PSNR: {:.6f} dB, SSIM: {:.6f}, SAM: {:.6f}, QI: {:.6f}, SCC: {:.6f}'.format(
            avg_psnr, avg_ssim, avg_sam, avg_qi, avg_scc))

        # Write averages to file
        f.write('PSNR: {:.6f} dB\n'.format(avg_psnr))
        f.write('SSIM: {:.6f}\n'.format(avg_ssim))
        f.write('SAM:  {:.6f}\n'.format(avg_sam))
        f.write('QI:   {:.6f}\n'.format(avg_qi))
        f.write('SCC:  {:.6f}\n'.format(avg_scc))

        # Write additional information
        f.write('\nNumber of processed images: {}\n'.format(len(img_list)))
        f.write('Number of valid metrics:\n')
        f.write('PSNR: {}\n'.format(len(PSNR_all)))
        f.write('SSIM: {}\n'.format(len(SSIM_all)))
        f.write('SAM:  {}\n'.format(len(SAM_all)))
        f.write('QI:   {}\n'.format(len(QI_all)))
        f.write('SCC:  {}\n'.format(len(SCC_all)))

    print(f'\nResults have been saved to {result_file}')

# [Rest of the functions remain unchanged]
def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_rgb_psnr(img1, img2):
    """calculate psnr among rgb channel, img1 and img2 have range [0, 255]
    """
    n_channels = img1.shape[2] if len(img1.shape) == 3 else 1
    sum_psnr = 0
    for i in range(n_channels):
        if n_channels == 1:
            this_psnr = calculate_psnr(img1, img2)
        else:
            this_psnr = calculate_psnr(img1[:, :, i], img2[:, :, i])
        sum_psnr += this_psnr
    return sum_psnr / n_channels

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(img1.shape[2]):
                ssims.append(ssim(img1[..., i], img2[..., i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

if __name__ == '__main__':
    main()
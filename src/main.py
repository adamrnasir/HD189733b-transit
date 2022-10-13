from astropy.io import fits
from astropy import wcs
from astropy.nddata.utils import Cutout2D
import numpy as np
import re
import glob
import matplotlib.pyplot as plt
import cv2

# Turn all the fits files from data/ into images and save them
# in data/images/
fit_files = glob.glob('data/*.fit')
fit_files.sort(key=lambda x: int(re.search(r'\d+', x).group(0)))

padding = (100, 100)

times = []
maxes = []
areas = []
brightnesses = []
area_times = []
nums = []

reference_image = cv2.imread('images/300.png', 0)
orb = cv2.ORB_create()
kp_ref, des_ref = orb.detectAndCompute(reference_image, None)

for f in fit_files:
    # Get a 1-3 digit number from the filename
    num = re.search(r'\d+', f).group(0)

    timestr = fits.getheader(f)['DATE-OBS'].split('T')[1]
    times.append(timestr)

    img_data = fits.getdata(f)
    maxes.append(np.max(img_data))

    # Get the median of the image
    median = np.mean(img_data)

    if int(num) % 100 == 0:
        print(f)

    # Get a bounding box around the matches in the current image
    try:
        # Normalize using cv2.normalize
        normalized = cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Binary threshold the image
        ret, thresh = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Feature match between the reference image and the current image
        # and draw the matches
        kp_cur, des_cur = orb.detectAndCompute(normalized, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_ref, des_cur)
        matches = sorted(matches, key=lambda x: x.distance)

        # Get the matches with x coordinate less than half the image width
        # This is to get rid of the matches that are on the other side of the image
        matches = [m for m in matches if kp_cur[m.trainIdx].pt[0] < normalized.shape[1] / 2]

        x1 = min([kp_cur[m.trainIdx].pt[0] for m in matches])
        x2 = max([kp_cur[m.trainIdx].pt[0] for m in matches])
        y1 = min([kp_cur[m.trainIdx].pt[1] for m in matches])
        y2 = max([kp_cur[m.trainIdx].pt[1] for m in matches])

        # Get center of bounding box
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        
        newxl = x - padding[0]
        newxr = x + padding[0]
        newyt = y - padding[1]
        newyb = y + padding[1]  

        # Use cutout2d to crop the image
        hdu = fits.open(f)
        w = wcs.WCS(hdu[0].header)
        cutout = Cutout2D(hdu[0].data, (x, y), (200, 200), wcs=w)
        hdu[0].data = cutout.data
        hdu[0].header.update(cutout.wcs.to_header())

        # if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
        #     continue

        hdu.writeto(f'cropped_data/{num}.fit', overwrite=True)

        # If any of x1, x2, y1, y2 are negative, skip this image

        # Crop the image to the bounding box
        # cropped = normalized[int(y1):int(y2), int(x1):int(x2)]

        # Save the cropped image to aligned/{num}.png
        # cv2.imwrite(f'images/{num}.png', cropped)

        continue

        # Get the area of the bounding box
        area = (x2 - x1) * (y2 - y1)

        # If area > 1000, skip this image
        if area > 5000 or area < 1000:
            continue

        # If the rectangle is too stretched out, skip this image
        if (x2 - x1) / (y2 - y1) > 1.5 or (x2 - x1) / (y2 - y1) < 0.5:
            continue

        # Get nonzero pixels in the bounding box from thresholded image
        # nonzero = np.nonzero(thresh[int(y1):int(y2), int(x1):int(x2)])

        # Get the average brightness of the nonzero pixels
        # brightness = np.mean(img_data[nonzero])


        # # Get average brightness within the bounding box
        brightness = np.mean(img_data[int(y1):int(y2), int(x1):int(x2)]) - median

        area_times.append(timestr)
        areas.append(area)
        brightnesses.append(brightness)
        nums.append(num)

        # Draw the bounding box on the image
        # cv2.rectangle(normalized, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # Save the image
        # cv2.imwrite(f'rects/{num}.png', normalized)


    except Exception:
        pass

# Plot max values over time, showing only every 100th label
# plt.plot(times, maxes)
# plt.xticks(times[::100], rotation=90)
# plt.show()

# Plot areas over time, showing only every 100th label
plt.plot(area_times, brightnesses)
plt.xticks(area_times[::100], rotation=90)
plt.show()

# Get the number of the 10 brightest images
brightest = np.argsort(brightnesses)[-10:]
brightest = [nums[i] for i in brightest]

# Print the 10 brightest images
for b in brightest:
    print(b)
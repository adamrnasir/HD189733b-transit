from astropy.io import fits
import numpy as np
import re
import glob
import matplotlib.pyplot as plt
import cv2

# Turn all the fits files from data/ into images and save them
# in data/images/
fit_files = glob.glob('data/CCD*.fit')
fit_files.sort(key=lambda x: int(re.search(r'\d+', x).group(0)))

padding = (10, 10)

times = []
maxes = []
areas = []
brightnesses = []
area_times = []

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
    median = np.median(img_data)

    if int(num) % 100 == 0:
        print(f)

    # Get a bounding box around the matches in the current image
    try:
        # Normalize using cv2.normalize
        normalized = cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Feature match between the reference image and the current image
        # and draw the matches
        kp_cur, des_cur = orb.detectAndCompute(normalized, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_ref, des_cur)
        matches = sorted(matches, key=lambda x: x.distance)

        # Get the matches with x coordinate less than half the image width
        # This is to get rid of the matches that are on the other side of the image
        matches = [m for m in matches if kp_cur[m.trainIdx].pt[0] < normalized.shape[1] / 2]

        x1 = min([kp_cur[m.trainIdx].pt[0] for m in matches]) - padding[0]
        x2 = max([kp_cur[m.trainIdx].pt[0] for m in matches]) + padding[0]
        y1 = min([kp_cur[m.trainIdx].pt[1] for m in matches]) - padding[1]
        y2 = max([kp_cur[m.trainIdx].pt[1] for m in matches]) + padding[1]
    
        # Get the area of the bounding box
        area = (x2 - x1) * (y2 - y1)

        # If area > 1000, pass
        if area > 2500 or area < 1000:
            continue

        # Get average brightness within the bounding box
        avg_brightness = np.mean(img_data[int(y1):int(y2), int(x1):int(x2)]) - median

        area_times.append(timestr)
        areas.append(area)
        brightnesses.append(avg_brightness)


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
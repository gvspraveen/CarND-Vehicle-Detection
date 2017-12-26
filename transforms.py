import numpy as np
import cv2
from skimage.feature import hog
from scipy.ndimage.measurements import label

# NOTE
# All these functions are very similar to what has been taught in class. The code will look similar to lesson code

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features

# Write a unified extract feature function to combine. This is based of `extract_features` in lessons.
# However, in lesson if only combines `spatial_features` and `hist_features`. I extended this method
# to also include hog features.
# Assumes images are read using `matplotlib.image.imread`
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                     spatial_transform=True, hist_transform=True, hist_bins=32,
                     hog_transform=True, pix_per_cell=8, cell_per_block=2, orient=9,
                     hog_channel='ALL'):
    # Create a list to append feature vectors to
    features = []
    for image in imgs:
        im_features = []
        feature_image = image
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        else: feature_image = np.copy(image)

        if spatial_transform:
            im_features.append(bin_spatial(feature_image, size=spatial_size))

        if hist_transform:
            im_features.append(color_hist(feature_image, nbins=hist_bins))

        if hog_transform:
            hog_features = []
            if hog_channel == 'ALL':
                for channel in range(3):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
            else:
                hog_features.append(get_hog_features(feature_image[:, :, hog_channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            im_features.append(np.ravel(hog_features))

        features.append(np.concatenate(im_features))
    return features

# Define a single function that can extract features using hog sub-sampling and make predictions
# This is similar to method explained in lesson videos
def find_cars(img, ystart, ystop,
              scale, svc, X_scaler, orient,
              pix_per_cell, cell_per_block,
              cspace, hog_channel='ALL',
              spatial_transform=True,
              spatial_size=(32, 32),
              hist_transform=True,
              hist_bins=32
              ):
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = img_tosearch

    all_boxes = []
    car_boxes = []

    if cspace != 'RGB':
        if cspace == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                     (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    if hog_channel == 'ALL':
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    else:
        hog_per_channel = get_hog_features(hog_channel, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            all_features = []
            # Extract HOG for this patch
            if hog_channel == 'ALL':
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog_per_channel[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            if spatial_transform:
                spatial_features = bin_spatial(subimg, size=spatial_size)
                all_features.append(spatial_features)

            if hist_transform:
                hist_features = color_hist(subimg, nbins=hist_bins)
                all_features.append(hist_features)

            all_features.append(hog_features)

            all_features_tuple = tuple(all_features)
            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack(all_features_tuple).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            xbox_left = np.int(xleft*scale)
            ytop_draw = np.int(ytop*scale)
            win_draw = np.int(window*scale)
            bounding_box = (
                (xbox_left, ytop_draw+ystart),
                (xbox_left+win_draw,ytop_draw+win_draw+ystart)
            )
            all_boxes.append(bounding_box)
            if test_prediction == 1:
                # cv2.rectangle(draw_img,(bounding_box[0][0], bounding_box[0][1]),
                #               (bounding_box[1][0], bounding_box[1][1]),(0,0,255),6)
                car_boxes.append(bounding_box)

    return all_boxes, car_boxes


def run_sliding_window(scales, img, final_svc, X_scaler, final_orient,
                       final_pix_per_cell, final_cell_per_block,
                       final_cspace, hog_channel='ALL',
                       spatial_transform=True,
                       spatial_size=(32, 32),
                       hist_transform=True,
                       hist_bins=32):
    # Given a series of scales, this method runs each one by find_cars method and collects
    # all bounding boxes
    final_all_boxes = []
    final_car_boxes = []
    for ystart, ystop, scale in scales:
        all_boxes, car_boxes = find_cars(img, ystart, ystop,
                                         scale, final_svc, X_scaler, final_orient,
                                         final_pix_per_cell, final_cell_per_block,
                                         final_cspace, hog_channel=hog_channel,
                                         spatial_transform=spatial_transform,
                                         spatial_size=spatial_size,
                                         hist_bins=hist_bins,
                                         hist_transform=hist_transform
                                         )
        final_all_boxes.extend(all_boxes)
        final_car_boxes.extend(car_boxes)

    return final_all_boxes, final_car_boxes

# This method walks through each bounding box and draws rectangle on image
def draw_bounding_boxes(img, bounding_boxes, color=(0,0,255)):
    draw_img = np.copy(img)
    for bounding_box in bounding_boxes:
        cv2.rectangle(draw_img,(bounding_box[0][0], bounding_box[0][1]),
                      (bounding_box[1][0], bounding_box[1][1]),color,6)
    return draw_img

# This is copied from lesson with slight modification
def add_heat(heatmap, bounding_boxes):
    # Iterate through list of bboxes
    draw_img = np.copy(heatmap)
    for bounding_box in bounding_boxes:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        draw_img[bounding_box[0][1]:bounding_box[1][1], bounding_box[0][0]:bounding_box[1][0]] += 1

    # Return updated heatmap
    return draw_img

# This is copied from lesson with slight modification (make a copy instead of overwriting image)
def apply_threshold(heatmap, threshold):
    draw_img = np.copy(heatmap)
    # Zero out pixels below the threshold
    draw_img[heatmap <= threshold] = 0
    # Return thresholded map
    return draw_img

def get_labels(heatmap):
    return label(heatmap)

# This code is copied from lesson with no modifications.
def draw_labeled_bboxes(draw_img, labels):
    img = np.copy(draw_img)
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 10)
    # Return the image
    return img
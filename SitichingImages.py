import numpy as np
import cv2


# 201735899 황수하
def sitich_img(ori_img1, ori_img2):
    img1 = cv2.cvtColor(ori_img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(ori_img2, cv2.COLOR_BGR2GRAY)

    detector = sift = cv2.xfeatures2d.SIFT_create()

    keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

    img_res_detect1 = cv2.drawKeypoints(img1, keypoints1, None)
    img_res_detect2 = cv2.drawKeypoints(img2, keypoints2, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)

    ratio_thresh = 0.4 # 0.4, 0.5
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    # draw matches

    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)

    cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    obj = np.empty((len(good_matches), 2), dtype=np.float32)
    scene = np.empty((len(good_matches), 2), dtype=np.float32)
    for k in range(len(good_matches)):
        obj[k, 0] = keypoints1[good_matches[k].queryIdx].pt[0]
        obj[k, 1] = keypoints1[good_matches[k].queryIdx].pt[1]
        scene[k, 0] = keypoints2[good_matches[k].trainIdx].pt[0]
        scene[k, 1] = keypoints2[good_matches[k].trainIdx].pt[1]

    H, _ = cv2.findHomography(scene, obj, cv2.RANSAC)

    output_img = cv2.warpPerspective(ori_img2, H,
                                     (ori_img1.shape[1] + ori_img2.shape[1],  # width
                                      max(ori_img1.shape[0], ori_img2.shape[0])))  # height

    output_img[0:ori_img1.shape[0], 0:ori_img1.shape[1]] = ori_img1

    return output_img

def crop(img):


    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img2=cv2.threshold(gray,0,255,cv2.THRESH_BINARY)[1]
    contours=cv2.findContours(img2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]

    x1=[]
    x2=[]
    y1=[]
    y2=[]
    for i in range(1,len(contours)):
        ret=cv2.boundingRect(contours[i])
        x1.append(ret[0])
        y1.append(ret[1])
        x2.append(ret[0]+ret[2])
        y2.append(ret[1]+ret[3])

    x1_min=min(x1)
    y1_min = min(x1)
    x2_max = max(x2)
    y2_max = max(y2)
    cv2.rectangle(img,(x1_min,y1_min),(x2_max,y2_max),(0,255,0),3)

    crop_img = img2[y1_min:y2_max,x1_min:x2_max]

    return img,crop_img


def ExtractImage(image):

    img_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 0, 255, 0)
    cnts,hierarchy =cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)

    mask = np.zeros(thresh.shape, dtype="uint8")
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    minRect = mask.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 0:
        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)

    cnts,hierarchy =cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    image = image[y:y + h, x:x + w]
    return image


if __name__ == "__main__":


#set 1
    img1 = cv2.imread('panorama/set1/07.png')
    img2 = cv2.imread('panorama/set1/06.png')
    img3 = cv2.imread('panorama/set1/08.png')

    img1 = cv2.flip(img1, 1)
    img2 = cv2.flip(img2, 1)

    img_left = sitich_img(img1, img2)
    img_left=cv2.flip(img_left,1)
    img_left= ExtractImage(img_left)

    cv2.flip(img_left,1)
    cv2.flip(img3,1)

    result = sitich_img(img_left,img3)
    cv2.flip(result,1)
    cv2.imshow("before remove",result)

    result=ExtractImage(result)
    cv2.imshow("after remove",result)
cv2.waitKey(0)









#set 2
    # img1 = cv2.imread('panorama/set2/001.png')
    # img2 = cv2.imread('panorama/set2/002.png')
    # img3 = cv2.imread('panorama/set2/003.png')
    # img4 = cv2.imread('panorama/set2/004.png')
    #
    # img1 = cv2.flip(img1,1)
    # img2 = cv2.flip(img2, 1)
    # img_left = sitich_img(img2, img1)
    #
    # img_left=cv2.flip(img_left,1)
    # img3=cv2.flip(img3,1)
    # img_right = sitich_img(img3,img_left)
    #
    # img_right = cv2.flip(img_right, 1)
    # img4 = cv2.flip(img4, 1)
    #
    # result=sitich_img(img4,img_right)
    #
    # cv2.imshow("result",result)
    #
    # result2=ExtractImage(result)
    # cv2.imshow("result2",result2)
    #
    #


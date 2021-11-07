from zxingcpp import read_barcode
import cv2
import numpy as np
class give_me_solution():
    def __init__(self,frame):
        self.frame  =  frame
        self.img = frame
        self.processed_frame = self.preprocess(frame)
        self.cluster = 3
        self.thresh = 100
#   Preprocessing the image using Adaptive Histogram Equalization method to modify the image contrast 
    def preprocess(self,image, clip_hist_percent=1):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                        #Gray scale
            hist = cv2.calcHist([gray],[0],None,[256],[0,256])                    #Histograms Gray scale image
            hist_size = len(hist)
            accumulator = []
            accumulator.append(float(hist[0]))  
            for index in range(1, hist_size):
                accumulator.append(accumulator[index -1] + float(hist[index]))
            maximum = accumulator[-1]
            clip_hist_percent *= (maximum/100.0)
            clip_hist_percent /= 2.0
            minimum_gray = 0
            while accumulator[minimum_gray] < clip_hist_percent:
                minimum_gray += 1
            maximum_gray = hist_size -1
            while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
                maximum_gray -= 1
            alpha = 255 / (maximum_gray - minimum_gray)
            beta = -minimum_gray * alpha
            result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            return result
        except:
            return image
#     Finding Barcodes using PYZBAR
    def find_barcode(self,img):
        barcodes = read_barcode(img)                                          #Decoding Processed Image
        flag = 0
        text = None
        if barcodes.valid:                                                           #If there is barcode 
            flag=1
            text = barcodes.text
        return self.img,flag,text                                                             #Returning detected results (result_image,True/False,Decoded Text)
    
#     Processing the preprocessed image using OTSU THRESH and detecting barcodes using PYZbar
    
    def OTSU_THRESH(self):
        gray = cv2.cvtColor(self.processed_frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        return self.find_barcode(thresh)
    
#     Processing the preprocessed image using canny edge detection with default thresh and detecting barcodes using PYZbar
    def CANNY(self):
        gray = cv2.cvtColor(self.processed_frame, cv2.COLOR_BGR2GRAY)
        im_bw = cv2.threshold(gray, self.thresh, 255, cv2.THRESH_BINARY)[1]
        im_bw_2 = cv2.threshold(im_bw, self.thresh, 255, cv2.THRESH_BINARY)[1]
        res = cv2.Canny(im_bw_2,200,200)+im_bw_2
        return self.find_barcode(res)
    
#     Processing the normal image using Normal Grayscale and detecting barcodes using PYZbar
    def Normal(self):
        gray = gray = cv2.cvtColor(self.processed_frame,cv2.COLOR_BGR2GRAY)
        return self.find_barcode(gray)
    
#     Detecting barcodes using DBR 
    def k_mean_clustered(self):
        image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        pixel_values = image.reshape((-1,3))
        pixel_values = np.float32(pixel_values)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 95,  0.85)
        k = 3
        retval, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape((image.shape))
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
        res = cv2.threshold(segmented_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        return self.find_barcode(res)

        
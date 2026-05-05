from cv2 import dilate, erode 

def morphology(gray_image, kernel):# .cpu().numpy()를 입력
    dilated_region = dilate(gray_image, kernel, iterations=5) #5
    result_gray_image = erode(dilated_region, kernel, iterations=2) #2
    #eroded = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=10) #closing
    # result_gray_image = gray_image * result_gray_image  # 픽셀 추출
    return result_gray_image


def morphology_visa(gray_image, kernel):# .cpu().numpy()를 입력
    # gray_image = (gray_image - gray_image.min()) /(gray_image.max()- gray_image.min())
    
    dilated_region = dilate(gray_image, kernel, iterations=3)
    result_gray_image = erode(dilated_region, kernel, iterations=1)
    
    # result_gray_image = (result_gray_image-result_gray_image.min()) / (result_gray_image.max() - result_gray_image.min())
    return result_gray_image

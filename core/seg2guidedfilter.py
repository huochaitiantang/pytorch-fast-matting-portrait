import cv2
import numpy as np
import argparse

def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Segmentation to Guided Filter Result')
    parser.add_argument('--imgDir', type=str, required=True, help="pictures directory")
    parser.add_argument('--predDir', type=str, required=True, help="prediction pictures directory")
    parser.add_argument('--saveDir', type=str, required=True, help="where filter result save to")
    parser.add_argument('--testList', type=str, required=True, help="list of images id")
    args = parser.parse_args()
    print(args)
    return args

def main():
    args = get_args()
    f = open(args.testList)
    names = f.readlines()
    print("Images Count: {}".format(len(names)))
    for name in names:
        img_name = args.imgDir + "/" + name.strip() + ".jpg"
        pred_name = args.predDir + "/" + name.strip() + ".jpg"
        des_name = args.saveDir + "/" + name.strip() + ".jpg"

        img = cv2.imread(img_name)
        pred = cv2.imread(pred_name)
        des = img.copy()        

        radius = 60
        eps = 1e-6
        GF = cv2.ximgproc.createGuidedFilter(img, radius, eps)
        GF.filter(pred, des)

        #indfg = np.where(des >= 128)
        #indbg = np.where(des <  128)
        #des[indfg] = 255
        #des[indbg] = 0

        print("Write to {}".format(des_name))
        cv2.imwrite(des_name, des)

if __name__ == "__main__":
    main()



import numpy as np
import argparse
import cv2

def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Eval Segmentation')
    parser.add_argument('--maskDir', type=str, required=True, help="mask pictures directory")
    parser.add_argument('--predDir', type=str, required=True, help="prediction pictures directory")
    parser.add_argument('--testList', type=str, required=True, help="list of images id")
    args = parser.parse_args()
    print(args)
    return args

def main():
    args = get_args()
    f = open(args.testList)
    names = f.readlines()
    print("Images Count: {}".format(len(names)))
    inters = 0.
    unions = 0.
    rights = 0.
    cnts = 0.
    for name in names:
        mask_name = args.maskDir + "/" + name.strip() + "_mask.jpg"
        pred_name = args.predDir + "/" + name.strip() + ".jpg"
        mask = cv2.imread(mask_name)[:,:,0] / 255
        pred = cv2.imread(pred_name)[:,:,0] / 255
        tmp = mask + pred
        inters += len(np.where(tmp == 2)[0])
        unions += len(np.where(tmp >= 1)[0])
        rights += ((mask == pred).sum())
        cnts += len(np.where(tmp >= 0)[0])

    mean_iou = inters / unions
    pixel_acc = rights / cnts
    print("Mean IoU: {:.4f}\nPixel Acc: {:.4f}".format(mean_iou, pixel_acc))   

if __name__ == "__main__":
    main()

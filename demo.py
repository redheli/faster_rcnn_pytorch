import cv2
import numpy as np
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

def test():
    import os
    # im_file = 'demo/004545.jpg'
    # im_file = 'data/WEIQIdevkit2017/WEIQI2017/JPEGImages/000004.jpg'
    # im_file = 'data/VOCdevkit2007/VOC2007/JPEGImages/009036.jpg'
    # im_file = '/media/longc/Data/data/2DMOT2015/test/ETH-Crossing/img1/000100.jpg'
    im_file = './IMG_2030.jpg'
    image = cv2.imread(im_file)

    # model_file = './VGGnet_fast_rcnn_iter_70000.h5'
    # model_file = '/media/longc/Data/models/faster_rcnn_pytorch3/faster_rcnn_100000.h5'
    # model_file = '/media/longc/Data/models/faster_rcnn_pytorch2/faster_rcnn_2000.h5'
    model_file = './models/saved_model_max2/faster_rcnn_2000.h5'
    detector = FasterRCNN()
    network.load_net(model_file, detector)
    detector.cuda()
    detector.eval()
    print('load model successfully!')

    # network.save_net(r'/media/longc/Data/models/VGGnet_fast_rcnn_iter_70000.h5', detector)
    # print('save model succ')

    t = Timer()
    t.tic()
    # image = np.zeros(shape=[600, 800, 3], dtype=np.uint8) + 255
    dets, scores, classes = detector.detect(image, 0.7)
    runtime = t.toc()
    print('total spend: {}s'.format(runtime))

    im2show = np.copy(image)
    img = mpimg.imread(im_file)
    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(img)
    # Create a Rectangle patch
    for i, det in enumerate(dets):
        w = det[2] - det[0]
        h = det[3] - det[1]
        rect = patches.Rectangle(det[0:2], w, h, linewidth=1, edgecolor='r', facecolor='none')
        # text
        plt.text(det[0], det[1], '%s: %.3f' % (classes[i], scores[i]))

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()
    print('aa')
    # for i, det in enumerate(dets):
    #     det = tuple(int(x) for x in det)
    #     cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 2)
    #     cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
    #                 1.0, (0, 0, 255), thickness=1)
    # cv2.imwrite(os.path.join('demo', 'out.jpg'), im2show)
    # cv2.imshow('demo', im2show)
    # cv2.waitKey(0)


if __name__ == '__main__':
    test()
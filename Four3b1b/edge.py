import cv2
import matplotlib.pyplot as plt


print('EDGE DETECTION Imported')


def simple_edge_detection(fpath, n_size=None, mini=100, maxi=200, save=False):
    # todo resize
    print('EDGE DETECTION')
    image = cv2.imread(fpath, 0)
    print('to resize')
    if n_size:
        image = cv2.resize(image, n_size)

    edges_detected = cv2.Canny(image, mini, maxi)
    images = [image, edges_detected]
    location = [121, 122]
    print('detected')
    for loc, edge_image in zip(location, images):
        plt.subplot(loc)
        plt.imshow(edge_image, cmap='gray')

    cv2.imwrite('edge_detected.png', edges_detected)
    if save:
        plt.savefig('edge_plot.png')
    plt.show()
    print('to return')
    return edges_detected


if __name__ == '__main__':
    bdata = simple_edge_detection(r'N:\PC stuff\Programs\Python\Fourier\test_image.jpg')
    print(bdata)

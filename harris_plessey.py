# by Paul-Ruben Schlumbom, psch250
import sys
import os
import numpy as np
from scipy import signal
from skimage.feature import peak_local_max
from matplotlib import pyplot as plt
from PIL import Image
from shutil import copy

report=False
display = False

def gaussian_function(x, sigma=1):
    mod = -1 if x < 0 else 1
    return np.e**(-(x**2)/(2*sigma**2))/(sigma * np.sqrt(2 * np.pi)) * mod

def gaussian_second_derivative_function(x, sigma=1):
    return (-x/(sigma**3*np.sqrt(2*np.pi)))*(np.e**(-(x**2)/(2*sigma**2)))

def gaussian_function_2D(x, y, sigma=1):
    return np.e ** (-((x ** 2) + (y ** 2)) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

def gaussian(n, sigma=1):
    return np.asarray([gaussian_function(x) for x in range(int(-n/2), int(n/2)+1)])

def gaussian_d2(n, sigma=1):
    return np.asarray([gaussian_second_derivative_function(x) for x in range(int(-n/2), int(n/2)+1)])

def gaussian_2D(n, sigma=1):
    kernel = np.ones((n, n))
    vals = list(range(int(-n/2), int(n/2)+1))
    for x in range(n):
        for y in range(n):
            kernel[x][y] = gaussian_function_2D(vals[x], vals[y])
    return kernel

def show_im(im, points, name=None):
    plt.clf()
    plt.imshow(im)
    plt.set_cmap('hot')
    plt.colorbar()
    plt.scatter(y=[p[0] for p in points], x=[p[1] for p in points], marker='x')
    if display:
        for p in points:
            plt.annotate('({},{})'.format(int(p[1]), int(p[0])), xy=(p[1],p[0]))
    if name: plt.savefig(name, dpi=900)
    #plt.show()
    plt.draw()

def get_fast_R(s_x2, s_y2, s_xy, k=0.04):
    Hs = np.asarray([[s_x2, s_xy],
                [s_xy, s_y2]])
    Hs = Hs.T # transpose for the purposes of the following operations
    lambdas = np.split(np.linalg.eig(Hs)[0], 2, axis=2)
    dets = lambdas[0]*lambdas[1]
    traces = lambdas[0]+lambdas[1]
    R_im = dets - (k*(traces**2))
    R_im = np.reshape(R_im, (R_im.shape[:2]))
    R_im = R_im.T # transpose again to get the original orientation back
    return R_im

def computeRMap(im, kernel_size=3, k=0.04):
    # Step 1. Compute  x  and  y  derivatives of image
    gdx = np.asarray([gaussian_d2(kernel_size)])
    gdx = np.vstack((gdx, gdx, gdx))
    if report: print("gdx: ", gdx)

    i_x = signal.convolve2d(im, gdx, 'same')
    i_y = signal.convolve2d(im, gdx.T, 'same')

    # Step 2. Compute products of derivatives at every pixel
    i_x2 = i_x * i_x
    i_y2 = i_y * i_y
    i_xy = i_x * i_y


    # Step 3. Compute the sums of the products of the derivatives at each pixel
    sum_filter_x = np.ones((kernel_size, kernel_size))
    sum_filter_x = gaussian_2D(kernel_size)
    sum_filter_x /= sum_filter_x[kernel_size // 2][kernel_size // 2]
    if report: print("sum_filter_x: ", sum_filter_x)

    s_x2 = signal.convolve2d(i_x2, sum_filter_x, 'same')
    s_y2 = signal.convolve2d(i_y2, sum_filter_x.T, 'same')
    s_xy = signal.convolve2d(i_xy, np.vstack((sum_filter_x, sum_filter_x, sum_filter_x)), 'same')

    # Step 4. Define the matrix  H  at each pixel ( x ,  y ), and 5: Compute the response of the detector at each pixel.
    return get_fast_R(s_x2, s_y2, s_xy, k)

def detect_points(im, R_im, thresh_divider=1000, kernel_size=3):
    thresh = np.amax(R_im) / thresh_divider

    R_t = (R_im > thresh) * R_im
    R_abs = (R_t > 0) * 255

    return peak_local_max(R_t, min_distance=kernel_size)

def prune_points(im, points, proxim_thresh, side, pos):
    """xmin, xmax = int(max(0, pos[0] - (side/2))), int(min(im.shape[0], pos[0] + (side/2)))
    ymin, ymax = int(max(0, pos[1] - (side / 2))), int(min(im.shape[1], pos[1] + (side / 2)))
    #minx, maxx = int(im.shape[0]*(box_prop/2)), int(im.shape[0]*(1 - box_prop/2))
    #miny, maxy = int(im.shape[1]*(box_prop/2)), int(im.shape[1]*(1 - box_prop/2))
    pruned_pts = []
    for point in points:
        if point[1] > xmin and point[1] < xmax and point[0] > ymin and point[0] < ymax:
            pruned_pts.append(point)"""
    pruned_pts = list(points)

    """done = False
    while not done:
        done = True
        covered = []
        subset = []
        for i in range(len(pruned_pts)):
            condensed = False
            for j in range(len(pruned_pts)):
                if i not in covered and j not in covered and i != j:
                    a, b = np.asarray(pruned_pts[i]), np.asarray(pruned_pts[j])
                    if np.sqrt(np.sum((a - b) ** 2)) < proxim_thresh:
                        subset.append((pruned_pts[i] + pruned_pts[j]) / 2)
                        covered.extend([i, j])
                        done = False
                        condensed = True
            if not condensed and i not in covered:
                subset.append(pruned_pts[i])
        pruned_pts = subset
        print(len(pruned_pts))"""

    covered = np.zeros((len(pruned_pts)))
    subset = []
    for i in range(len(pruned_pts)):
        collected = []
        if covered[i] != 1:
            for j in range(len(pruned_pts)):
                if covered[j] != 1:
                    a, b = np.asarray(pruned_pts[i]), np.asarray(pruned_pts[j])
                    if np.sqrt(np.sum((a - b) ** 2)) < proxim_thresh:
                        collected.append(pruned_pts[j])
                        covered[j] = 1
        if len(collected) > 0:
            collected.append(pruned_pts[i])
            covered[i] = 1
            subset.append(sum(collected) / len(collected))
        elif covered[i] != 1:
            covered[i] = 1
            subset.append(pruned_pts[i])
    pruned_pts = subset


    """col = True
    while col == True:
        col = False
        for i in range(len(pruned_pts)):
            if i < len(pruned_pts):
                for j in range(len(pruned_pts)):
                    dropped = []
                    if j < len(pruned_pts):
                        a, b = np.asarray(pruned_pts[i]), np.asarray(pruned_pts[j])
                        if a[0] != b[0] and a[1] != b[1]:
                            if np.sqrt(np.sum((a-b)**2)) < proxim_thresh:
                                pruned_pts[i] = (pruned_pts[i] + pruned_pts[j]) / 2
                                dropped.append(j)
                                col = True"""
    """coll = True
    while coll:
        for point in pruned_pts:
            for otherpt in pruned_pts:"""
    return pruned_pts


if __name__ == "__main__":
    imagename = sys.argv[1]
    outname = sys.argv[2]
    imagepath =  imagename
    quit = False

    while not quit:
        print("Processing image...")
        imagepath = imagename
        #imagepath = imagename
        im = Image.open(imagepath).convert('L')
        im = np.asarray(im)


        k = 0.04
        thresh_divider = 1000
        kernel_size = 3
        dist_thresh = 0
        R_im = computeRMap(im, k=k)

        plt.ion()

        print("R-map calculated. Looking for points...")
        cont = True
        while cont:
            print("------\nk: {}\nthreshold divider: {}\nkernel size: {}\n------".format(k, thresh_divider, kernel_size))
            points = detect_points(im, R_im, thresh_divider=thresh_divider, kernel_size=kernel_size)

            print("{} points detected.".format(len(points)))
            if dist_thresh > 0:
                points = prune_points(im, points, dist_thresh, 1500, (1400,1000))
                print("{} condensed points detected.".format(len(points)))
            #plt.imshow(im)
            #plt.scatter(y=[p[0] for p in points], x=[p[1] for p in points])
            #plt.show()

            show_im(im, points)
            if "y" in input("Keep this result? y/n: "):
                outpath = 'results/'+outname+'/'
                if not os.path.exists(outpath[:-1]):
                    os.makedirs(outpath[:-1])

                #plt.imshow(im)
                #plt.scatter(y=[p[0] for p in points], x=[p[1] for p in points])
                #plt.savefig(outpath + outname + '.png', dpi=900)
                show_im(im, points, name=outpath + outname + '_detected.png')
                copy(imagepath, outpath)

                # create & save new csv file
                # note that for some reason, peak_local_max returns coordinates in format (y, x)
                out_lines = ["\n{},{}".format(v, im.shape[0] - u) for (u, v) in points]
                with open(outpath + outname + '_predicted-points.csv', 'w') as f:
                    f.write("k,{},threshdiv,{},kernel,{}\n".format(k, thresh_divider, kernel_size))
                    f.write("u,v")
                    for line in out_lines:
                        # print(line)
                        f.write(line)
                print("Saved image {} and CSV file {}.".format(outname + '_detected.png', outname + '_predicted-points.csv'))
                cont = False
                break
            else:
                command = input("Enter comma-separated modifications prefixed by variable ID, e.g. A{},B{},C{}\nA: k\nB: threshold divider\nC: kernel size\nD: threshold proximity for point pruning (0 for no pruning)\nq: quit\n".format(k, thresh_divider, kernel_size))
                commands = command.split(',')
                for mod in commands:
                    if 'A' in mod:
                        print("Recalculating R map...")
                        k = float(mod[1:])
                        R_im = computeRMap(im, kernel_size=kernel_size, k=k)
                    elif 'B' in mod:
                        thresh_divider = float(mod[1:])
                    elif 'C' in mod:
                        print("Recalculating R map...")
                        kernel_size = int(mod[1:])
                        R_im = computeRMap(im, kernel_size=kernel_size, k=k)
                    elif 'D' in mod:
                        dist_thresh = float(mod[1:])
                    else: cont = False

        if 'n' in input("Process another image? y/n: "):
            quit = True
        else:
            imagename = input("Enter new image path: ")
            outname = input("Enter new output name: ")
    print("Goodbye!")
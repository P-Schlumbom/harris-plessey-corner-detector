# from https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_fast/py_fast.html
# by Paul-Ruben Schlumbom, psch250
import sys
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from shutil import copy

report=False
display = False

def show_im(im, points, name=None):
    plt.clf()
    #im2 = im
    #im2 = cv2.drawKeypoints(im, points, im2, color=(255, 0, 0))
    plt.imshow(im)
    plt.set_cmap('hot')
    plt.scatter(y=[p[1] for p in points], x=[p[0] for p in points], marker='x')
    if display:
        for p in points:
            plt.annotate('({},{})'.format(p[1], p[0]), xy=(p[1],p[0]))
    if name: plt.savefig(name, dpi=900)
    plt.draw()

def prune_points(im, points, proxim_thresh, side, pos):
    pruned_pts = list(points)

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
        im = cv2.imread(imagepath, 0)


        threshold = 20
        nonmax_suppression = 1
        neighbourhood = 2
        dist_thresh = 0

        # Initiate FAST object with default values
        fast = cv2.FastFeatureDetector_create()
        fast.setThreshold(threshold)
        fast.setNonmaxSuppression(nonmax_suppression)
        fast.setType(neighbourhood)

        plt.ion()

        cont = True
        while cont:
            print("------\nthreshold: {}\nnonmax suppression: {}\nneighbourhood: {}\n-----".format(threshold, nonmax_suppression, neighbourhood))
            points = fast.detect(im)
            points = [point.pt for point in points]

            print("{} points detected.".format(len(points)))
            if dist_thresh > 0:
                points = prune_points(im, points, dist_thresh, 1500, (1400,1000))
                print("{} condensed points detected.".format(len(points)))

            show_im(im, points)
            if "y" in input("Keep this result? y/n: "):
                outpath = 'results-fast/'+outname+'/'
                if not os.path.exists(outpath[:-1]):
                    os.makedirs(outpath[:-1])

                show_im(im, points, name=outpath + outname + '_detected.png')
                copy(imagepath, outpath)

                # create & save new csv file
                out_lines = ["\n{},{}".format(u, im.shape[0] - v) for (u, v) in points]
                with open(outpath + outname + '_predicted-points.csv', 'w') as f:
                    f.write("threshold,{},nonmax_suppression,{},neighbourhood,{}\n".format(threshold, nonmax_suppression, neighbourhood))
                    f.write("u,v")
                    for line in out_lines:
                        # print(line)
                        f.write(line)
                print("Saved image {} and CSV file {}.".format(outname + '_detected.png', outname + '_predicted-points.csv'))
                cont = False
                break
            else:
                command = input("Enter comma-separated modifications prefixed by variable ID, e.g. A{},B{},C{}\nA: threshold\nB: nonmax_suppression\nC: neighbourhood\nD: threshold proximity for point pruning (0 for no pruning)\nq: quit\n".format(threshold, nonmax_suppression, neighbourhood))
                commands = command.split(',')
                for mod in commands:
                    if 'A' in mod:
                        threshold = int(mod[1:])
                        fast.setThreshold(threshold)
                    elif 'B' in mod:
                        nonmax_suppression = int(mod[1:])
                        fast.setNonmaxSuppression(nonmax_suppression)
                    elif 'C' in mod:
                        neighbourhood = int(mod[1:])
                        fast.setType(neighbourhood)
                    elif 'D' in mod:
                        dist_thresh = float(mod[1:])
                    else: cont = False

        if 'n' in input("Process another image? y/n: "):
            quit = True
        else:
            imagename = input("Enter new image path: ")
            outname = input("Enter new output name: ")
    print("Goodbye!")
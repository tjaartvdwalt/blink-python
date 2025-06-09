import cv2 as cv
import numpy as np


def run(blink_ear_file: str, non_blink_ear_file: str, model_file: str):
    print("init...")
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setC(10)
    # svm.setTermCriteria((cv.TERM_CRITERIA_EPS, 100, 1e-6))
    svm.setTermCriteria((cv.TERM_CRITERIA_EPS, 10000, 1e-6))

    training_data = []
    labels = []

    for file, label in [(blink_ear_file, 1), (non_blink_ear_file, -1)]:
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                if not line.startswith("#"):
                    list = line.split(" ")
                    training_list = [float(i) for i in list]
                    training_data.append(training_list)
                    labels.append(label)

    print(training_data)
    print(labels)
    print("training svm...")

    # mat = np.array(training_data, dtype=np.float32),
    # for el in mat:
    #     print(el)
    #     print("")

    svm.trainAuto(
        np.array(training_data, dtype=np.float32),
        cv.ml.ROW_SAMPLE,
        np.array(labels),
        balanced=True,
    )

    svm.save(model_file)


# svm = cv.ml.SVM_load("../svm.xml")
#
# sampleMat = np.matrix(
#     [[1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]],
#     dtype=np.float32,
# )
# response = svm.predict(sampleMat)
#
# print(response)


# # Data for visual representation
# width = 512
# height = 512
# image = np.zeros((height, width, 3), dtype=np.uint8)
# # Show the decision regions given by the SVM
# green = (0, 255, 0)
# blue = (255, 0, 0)
# for i in range(image.shape[0]):
#     for j in range(image.shape[1]):
#         sampleMat = np.matrix([[j, i]], dtype=np.float32)
#         response = svm.predict(sampleMat)[1]
#         if response == 1:
#             image[i, j] = green
#         elif response == -1:
#             image[i, j] = blue
# # Show the training data
# thickness = -1
# cv.circle(image, (501, 10), 5, (0, 0, 0), thickness)
# cv.circle(image, (255, 10), 5, (255, 255, 255), thickness)
# cv.circle(image, (501, 255), 5, (255, 255, 255), thickness)
# cv.circle(image, (10, 501), 5, (255, 255, 255), thickness)
#
# # Show support vectors
# thickness = 2
# sv = svm.getUncompressedSupportVectors()
# for i in range(sv.shape[0]):
#     cv.circle(image, (int(sv[i, 0]), int(sv[i, 1])), 6, (128, 128, 128), thickness)
# cv.imwrite("result.png", image)  # save the image
# cv.imshow("SVM Simple Example", image)  # show it to the user
# cv.waitKey()

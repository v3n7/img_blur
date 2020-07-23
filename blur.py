from numba import njit
import cv2
import numpy


def get_color_array(color_id):
    if color_id == 0:
        return numpy.array([1, 0, 0])
    elif color_id == 1:
        return numpy.array([0, 1, 0])
    else:
        return numpy.array([0, 0, 1])


@njit
def mult(matr, core):
    result = 0
    for i in range(len(matr)):
        for j in range(len(matr)):
            result += matr[i][j] * core[i][j]
    return result


def bokeh(image, core, file_name):
    conv_core = core / MAX_BRIGHTNESS
    shift_x = conv_core.shape[0] // 2
    shift_y = conv_core.shape[1] // 2
    size_x = image.shape[0]
    size_y = image.shape[1]
    image2 = numpy.zeros(
        (size_x + shift_x * 2, size_y + shift_y * 2, 3), dtype=numpy.uint8)
    img_slice_x = slice(shift_x, (size_x + shift_x))
    img_slice_y = slice(shift_y, (size_y + shift_y))
    image2[img_slice_x, img_slice_y, 0:3] = image
    image2 = image2 / MAX_BRIGHTNESS
    slice_z = slice(0, 3)
    for i in range(shift_x, size_x + shift_x):
        for j in range(shift_y, size_y + shift_y):
            slice_x = slice(i - shift_x, (i + shift_x + 1))
            slice_y = slice(j - shift_y, (j + shift_y + 1))
            matrix = image2[slice_x, slice_y, slice_z]
            for color in range(3):
                result = mult(matrix[:, :, color], conv_core[:, :, color])
                image2[i][j][color] = result / \
                    numpy.sum(conv_core[:, :, color])

    image2 = image2[img_slice_x, img_slice_y, slice_z]
    image2 = image2 * MAX_BRIGHTNESS
    image2.astype(numpy.uint8)
    cv2.imwrite(file_name, image2)


MAX_BRIGHTNESS = 255


def main():
    files = []
    files.append(
        {
            "image": cv2.imread(r"C:\Python Scripts\Blur\night_city_lights_512.jpg"),
            "core": cv2.imread(r"C:\Python Scripts\Blur\conv_core.png"),
            "result_name": "blur_1.jpg"})
    files.append(
        {
            "image": cv2.imread(r"C:\Python Scripts\Blur\rose.png"),
            "core": cv2.imread(r"C:\Python Scripts\Blur\conv_core_2.png"),
            "result_name": "blur_2.jpg"})
    files.append(
        {
            "image": cv2.imread(r"C:\Python Scripts\Blur\moscow.png"),
            "core": cv2.imread(r"C:\Python Scripts\Blur\conv_core_3.png"),
            "result_name": "blur_3.jpg"})

    for arr_file in files:
        name = arr_file["result_name"]
        result_name = f"C:\\Python Scripts\\Blur\\{name}"
        bokeh(arr_file["image"], arr_file["core"], result_name)


if __name__ == "__main__":
    main()

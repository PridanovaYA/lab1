import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow


def get_channel_rgb(image, channel):
    b, g, r = cv2.split(image)
    if channel == 'blue':
        return b
    if channel == 'green':
        return g
    if channel == 'red':
        return r


def get_channel_cmy(image, channel):
    #b, g, r = cv2.split(image)
    img = image.astype(np.float64) / 255.
    K = 1 - np.max(img, axis=2)
    if channel == 'cyan':
       C = (1 - img[..., 2] - K) / (1 - K)
       return C
    if channel == 'magenta':
        M = (1 - img[..., 1] - K) / (1 - K)
        return M
    if channel == 'yellow':
        Y = (1 - img[..., 0] - K) / (1 - K)
        return Y



def get_plane(channel_image, plane_num):
    return channel_image | (2 ** (plane_num - 1))


def encode_svi1(image, watermark, channel_color, bit_num):
    # num_for_clear_bit_plate = 255 - (2 ** (bit_num - 1))

    prepared_watermark = (watermark * (2 ** (bit_num - 1))).astype(np.uint8)
    watermark_channel = get_channel_rgb(prepared_watermark, channel_color)
    # cv2.imshow("watermark",  watermark_channel)
    # prepared_image = (image * (2 ** (bit_num - 1))).astype(np.uint8)
    #cv2.imshow("test_2", prepared_image)
    prepared_image_channel = get_channel_rgb(image, channel_color)
    #cv2.imshow("test_3", prepared_image_channel)
    # image_with_empty_bit = get_channel_rgb(image, channel_color) & num_for_clear_bit_plate # очищаем нужную битовую плоскость
    result_image = (prepared_image_channel + watermark_channel) # 3.4
    # cv2.imshow("test_4", result_image)

    r = get_channel_rgb(image, 'red')
    g = get_channel_rgb(image, 'green')
    b = get_channel_rgb(image, 'blue')

    # cv2.imshow("test_4", result_image)
    if channel_color == 'blue':
        return cv2.merge([result_image, g, r])
    if channel_color == 'red':
        return cv2.merge([b, g, result_image])
    if channel_color == 'green':
        return cv2.merge([b, result_image, r])


def decode_svi1(image, encoded_image, channel_color, bit_num):
    encoded_encoded_image_channel = get_channel_rgb(encoded_image, channel_color)
    prepared_image_channel = get_channel_rgb(image, channel_color)
    prepared_image = (prepared_image_channel & (2 ** (bit_num - 1))).astype(np.uint8)
    prepared_encoded_image = (encoded_encoded_image_channel & (2 ** (bit_num - 1))).astype(np.uint8)
    watermark = prepared_encoded_image ^ prepared_image

    result_image = encoded_encoded_image_channel ^ watermark

    watermark = (watermark * 255).astype(np.uint8)
    # cv2.imshow("watermark", watermark)

    r = get_channel_rgb(encoded_image, 'red')
    g = get_channel_rgb(encoded_image, 'green')
    b = get_channel_rgb(encoded_image, 'blue')

    if channel_color == 'blue':
        return watermark, cv2.merge([result_image, g, r])
    if channel_color == 'red':
        return watermark, cv2.merge([b, g, result_image])
    if channel_color == 'green':
        return watermark, cv2.merge([b, result_image, r])


def encode_svi4(image, watermark, channel_color, delta):
    h, w, channels = image.shape
    # noise = np.empty((h, w), dtype="uint8")
    # cv2.randn(noise, 0, delta - 1)
    # cv2.imshow("Noise", noise)

    extracted_channel = get_channel_cmy(image, channel_color)
    noise = extracted_channel % delta
    # cv2.imshow("Noise", noise * 255)
    binary_watermark = get_channel_cmy(watermark, channel_color)
    changed_channel = (extracted_channel // (2 * delta) * (2 * delta)) + binary_watermark * delta + noise  # 3.10

    # r = get_channel(image, 'red')
    # g = get_channel(image, 'green')
    # b = get_channel(image, 'blue')
    img_1 = image.astype(np.float64) / 255.
    K = 1 - np.max(img_1, axis=2)
    C = get_channel_cmy(image, 'cyan')
    M = get_channel_cmy(image, 'magenta')
    Y = get_channel_cmy(image, 'yellow')

    if channel_color == 'cyan':
        return noise, (np.dstack((changed_channel,M,Y,K)) * 255).astype(np.uint8)
    if channel_color == 'yellow':
        return noise, cv2.merge([C, M, changed_channel])
    if channel_color == 'magenta':
        return noise, cv2.merge([C, changed_channel, Y])


def decode_svi4(encoded_image, original_image, noise, channel_color, delta):
    encoded_image_channel = get_channel_cmy(encoded_image, channel_color)
    original_image_channel = get_channel_cmy(original_image, channel_color)
    return (encoded_image_channel - noise - (original_image_channel // (2 * delta) * 2 * delta)) / delta


def print_images(first, second, third, titles):
    fig = plt.figure(figsize=(12, 5))
    fig.add_subplot(1, 3, 1)
    imshow(cv2.cvtColor(first, cv2.COLOR_BGR2RGB))
    fig.suptitle(titles[0])
    plt.title(titles[1])

    fig.add_subplot(1, 3, 2)
    imshow(cv2.cvtColor(second, cv2.COLOR_BGR2RGB))
    plt.title(titles[2])

    fig.add_subplot(1, 3, 3)
    imshow(cv2.cvtColor(third, cv2.COLOR_BGR2RGB))
    plt.title(titles[3])

    plt.show()


if __name__ == '__main__':
    baboon = cv2.imread('baboon.tif')
    ornament = cv2.imread('ornament.tif')
    mickey = cv2.imread('mickey.tif')
    # cv2.imshow("Original", baboon)

    svi_1_result = encode_svi1(baboon, ornament, 'green', 2)
    print_images(baboon, ornament, svi_1_result, ['встраивание 1', 'контейнер', 'цвз', 'результат'])
    #
    svi_1_result = encode_svi1(svi_1_result, mickey, 'blue', 1)
    print_images(baboon, mickey, svi_1_result, ['встраивание 2', 'контейнер', 'цвз', 'результат'])
    #
    watermark_1, svi_1_decode_1 = decode_svi1(baboon, svi_1_result, 'green', 2)
    print_images(baboon, watermark_1, svi_1_decode_1, ['извлечение 1', 'контейнер', 'цвз', 'результат'])
    #
    watermark_2, svi_1_decode_2 = decode_svi1(baboon, svi_1_result, 'blue', 1)
    print_images(baboon, watermark_2, svi_1_decode_2, ['извлечение 2', 'контейнер', 'цвз', 'результат'])


    cv2.waitKey(0)
    VAR = 6
    DELTA = 4 + (4 * VAR) % 3

    result_noise, svi_4_result = encode_svi4(baboon, ornament, 'cyan', DELTA)
    #print_images(baboon, result_noise * 255, svi_4_result, ['встраивание 3', 'контейнер', 'цвз', 'результат'])
    svi_4_decode = decode_svi4(svi_4_result, baboon, result_noise, 'cyan', DELTA)
    #print_images(baboon, result_noise, svi_4_decode, ['извлечение 3', 'контейнер', 'цвз', 'результат'])

    cv2.imshow("Original", baboon)
    cv2.imshow("SVI-4 Encoded", svi_4_result)
    cv2.imshow("SVI-4 Decoded", svi_4_decode)

    cv2.waitKey(0)

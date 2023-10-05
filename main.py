import cv2
import numpy as np


def get_channel_rgb(image, channel):
    b, g, r = cv2.split(image)
    if channel == 'blue':
        return b
    if channel == 'green':
        return g
    if channel == 'red':
        return r


def get_channel_cmy(image, channel):
    b, g, r = cv2.split(image)
    if channel == 'cyan':
        return 1 - r
    if channel == 'magenta':
        return 1 - g
    if channel == 'yellow':
        return 1 - b


def get_plane(channel_image, plane_num):
    return channel_image | (2 ** (plane_num - 1))


def encode_svi1(image, watermark, channel_color, bit_num):
    # num_for_clear_bit_plate = 255 - (2 ** (bit_num - 1))

    prepared_watermark = (watermark * (2 ** (bit_num - 1))).astype(np.uint8)
    watermark_channel = get_channel_rgb(prepared_watermark, channel_color)
    cv2.imshow("test",  watermark_channel)
    prepared_image = (image * (2 ** (bit_num - 1))).astype(np.uint8)
    # image_with_empty_bit = get_channel_rgb(image, channel_color) & num_for_clear_bit_plate # очищаем нужную битовую плоскость
    result_image = get_channel_rgb(prepared_image, channel_color) | watermark_channel  # 3.4

    r = get_channel_rgb(baboon, 'red')
    g = get_channel_rgb(baboon, 'green')
    b = get_channel_rgb(baboon, 'blue')

    if channel_color == 'blue':
        return cv2.merge([result_image, g, r])
    if channel_color == 'red':
        return cv2.merge([b, g, result_image])
    if channel_color == 'green':
        return cv2.merge([b, result_image, r])


def decode_svi1(image, encoded_image, channel_color, bit_num):
    prepared_encoded_image = (encoded_image * (2 ** (bit_num - 1))).astype(np.uint8)
    encoded_encoded_image_channel = get_channel_rgb(prepared_encoded_image, channel_color)
    prepared_image = (image * (2 ** (bit_num - 1))).astype(np.uint8)
    prepared_image_channel = get_channel_rgb(prepared_image, channel_color)

    result = encoded_encoded_image_channel | prepared_image_channel

    return result


def encode_svi4(image, watermark, channel_color, delta):
    h, w, channels = image.shape
    noise = np.empty((h, w), dtype="uint8")
    # cv2.randn(noise, 0, delta - 1)
    cv2.imshow("Noise", noise)

    extracted_channel = get_channel_cmy(image, channel_color)
    noise = extracted_channel % delta
    cv2.imshow("Noise", noise)
    binary_watermark = get_channel_cmy(watermark, channel_color)

    changed_channel = (extracted_channel // (2 * delta) * (2 * delta)) + binary_watermark * delta + noise  # 3.10

    # r = get_channel(image, 'red')
    # g = get_channel(image, 'green')
    # b = get_channel(image, 'blue')
    c = get_channel_cmy(image, 'cyan')
    m = get_channel_cmy(image, 'magenta')
    y = get_channel_cmy(image, 'yellow')

    if channel_color == 'cyan':
        return noise, cv2.merge([changed_channel, m, y])
    if channel_color == 'yellow':
        return noise, cv2.merge([c, m, changed_channel])
    if channel_color == 'magenta':
        return noise, cv2.merge([c, changed_channel, y])


def decode_svi4(encoded_image, original_image, noise, channel_color, delta):
    encoded_image_channel = get_channel_cmy(encoded_image, channel_color)
    original_image_channel = get_channel_cmy(original_image, channel_color)
    return (encoded_image_channel - noise - (original_image_channel // (2 * delta) * 2 * delta)) / delta


if __name__ == '__main__':
    baboon = cv2.imread('baboon.tif')
    ornament = cv2.imread('ornament.tif')
    mickey = cv2.imread('mickey.tif')
    cv2.imshow("Original", baboon)

    svi_1_result = encode_svi1(baboon, ornament, 'green', 2)
    #svi_1_result = encode_svi1(baboon, mickey, 'blue', 1)

    svi_1_decode_1 = decode_svi1(baboon, svi_1_result, 'green', 2)
    #svi_1_decode_2 = decode_svi1(baboon, svi_1_result, 'blue', 1)

    cv2.imshow("Original", baboon)
    cv2.imshow("SVI-1 Encoded", svi_1_result)
    cv2.imshow("SVI-1 Decoded", svi_1_decode_1)

    cv2.waitKey(0)
    VAR = 6
    DELTA = 4 + (4 * VAR) % 3

    # result_noise, svi_4_result = encode_svi4(baboon, ornament, 'cyan', DELTA)
    # svi_4_decode = decode_svi4(svi_4_result, baboon, result_noise, 'cyan', DELTA)

    # cv2.imshow("Original", baboon)
    # cv2.imshow("SVI-4 Encoded", svi_4_result)
    # cv2.imshow("SVI-4 Decoded", svi_4_decode)

    cv2.waitKey(0)

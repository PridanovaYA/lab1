import cv2
import numpy as np


def get_channel(image, channel):
    b, g, r = cv2.split(image)
    if channel == 'blue':
        return b
    if channel == 'green':
        return g
    if channel == 'red':
        return r
    if channel == 'cyan':
        return 1 - r


def get_plane(channel_image, plane_num):
    return channel_image & (2 ** (plane_num - 1))


def encode_svi1(image, watermark, channel_color, bit_num):
    num_for_clear_bit_plate = 255 - (2 ** (bit_num - 1))

    prepared_watermark = ((watermark / 255) * (2 ** (bit_num - 1))).astype(np.uint8)
    watermark_channel = get_channel(prepared_watermark, channel_color)

    image_with_empty_bit = get_channel(image, channel_color) & num_for_clear_bit_plate

    result_image = image_with_empty_bit | watermark_channel

    r = get_channel(baboon, 'red')
    g = get_channel(baboon, 'green')
    b = get_channel(baboon, 'blue')

    if channel_color == 'blue':
        return cv2.merge([result_image, g, r])
    if channel_color == 'red':
        return cv2.merge([b, g, result_image])
    if channel_color == 'green':
        return cv2.merge([b, result_image, r])


def decode_svi1(encoded_image, channel_color, bit_num):
    encoded_image_channel = get_channel(encoded_image, channel_color)
    return get_plane(encoded_image_channel, bit_num)


def encode_svi4(image, watermark, channel_color, delta):
    h, w, channels = image.shape
    noise = np.empty((h, w), dtype="uint8")
    #cv2.randn(noise, 0, delta - 1)
    cv2.imshow("Noise", noise)

    extracted_channel = get_channel(image, channel_color)
    noise = extracted_channel % delta
    cv2.imshow("Noise", noise)
    binary_watermark = get_channel(watermark, channel_color)

    changed_channel = (extracted_channel // (2 * delta) * (2 * delta)) + binary_watermark * delta + noise # 3.10

    r = get_channel(image, 'red')
    g = get_channel(image, 'green')
    b = get_channel(image, 'blue')
    c = get_channel(image, 'cyan')

    if channel_color == 'blue':
        return noise, cv2.merge([changed_channel, g, r])
    if channel_color == 'red':
        return noise, cv2.merge([b, g, changed_channel])
    if channel_color == 'green':
        return noise, cv2.merge([b, changed_channel, r])
    if channel_color == 'cyan':
        return noise, cv2.merge([b, g, changed_channel])


def decode_svi4(encoded_image, original_image, noise, channel_color, delta):
    encoded_image_channel = get_channel(encoded_image, channel_color)
    original_image_channel = get_channel(original_image, channel_color)
    return (encoded_image_channel - noise - (original_image_channel // (2 * delta) * 2 * delta)) / delta


if __name__ == '__main__':
    baboon = cv2.imread('baboon.tif')
    ornament = cv2.imread('ornament.tif')
    cv2.imshow("Original", baboon)

    svi_1_result = encode_svi1(baboon, ornament, 'green', 2)
    svi_1_decode = decode_svi1(svi_1_result, 'blue', 1)

    cv2.imshow("Original", baboon)
    cv2.imshow("SVI-1 Encoded", svi_1_result)
    cv2.imshow("SVI-1 Decoded", svi_1_decode)

    cv2.waitKey(0)
    VAR = 6
    DELTA = 4 + (4 * VAR) % 3

    result_noise, svi_4_result = encode_svi4(baboon, ornament, 'cyan', DELTA)
    svi_4_decode = decode_svi4(svi_4_result, baboon, result_noise, 'cyan', DELTA)

    cv2.imshow("Original", baboon)
    cv2.imshow("SVI-4 Encoded", svi_4_result)
    cv2.imshow("SVI-4 Decoded", svi_4_decode)

    cv2.waitKey(0)
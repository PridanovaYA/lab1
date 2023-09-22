import cv2
import numpy as np


def lsb_embed(C, b, seed: int = -1):
    if b.size > C.size:  # C.size[0] * C.size[1]:
        print("bad data")
        return -1
    C_flattend = C.flatten() #осталась одна битовая плоскость
    C_flattend %= 2
    if (seed <0):
        C_flattend[: b.size] = b
    else:
        np.random.seed(seed)
        indeces = np.random.choice(np.arrange (0,  C_flattend.size), replace = False, size = b.size)
        C_flattend [indeces] = b

    C_cleaned =  (C // 2)*2 # очистим
    return C_cleaned + C_flattend.reshape(C.shape[0], C.shape[1])
    #print(C_flattend)


def text_to_bin(text):
    return np.unpackbits(np.fromstring(text, dtype = np.uint8))


def lsb_extract(C, Nb, seed: int = -1):
    if Nb > C.size:  # C.size[0] * C.size[1]:
        print("bad data")
        return -1
    C_flattend = C.flatten() #осталась одна битовая плоскость
    C_flattend %= 2
    if seed <0:
        return C_flattend[:Nb]
    else:
        np.random.seed(seed)
        indeces = np.random.choice(np.arrange(0, C_flattend.size), replace=False, size=b.Nb)
        return C_flattend[indeces]



if __name__ == '__main__':
    Nb = 10
    b = text_to_bin("Vety important text")
    C = cv2.imread('bridge.tif') # изображение должно быть на два канала, а сейчас три
    CW = lsb_embed(C, b)

    b_ext = lsb_extract(CW, len(b))
    ber = (b^b_ext).sum() / len(b)

    Nb = np.floor (C.size *0.2.astype(np.int32))
    b = np.random.randint(0,2, Nb)
    #b = np.zeros (1, Nb)
    CW_1 = lsb_embed (C, b)
    CW_2 = lsb_embed(C, b, 2)
    image_1 = CW_1%2
    image_2 = CW_2 % 2
    cv2.imshow()
import array
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import array
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# TODO: 뭐에 쓰는 값
crc_table = [
    0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50a5, 0x60c6, 0x70e7,
    0x8108, 0x9129, 0xa14a, 0xb16b, 0xc18c, 0xd1ad, 0xe1ce, 0xf1ef,
    0x1231, 0x0210, 0x3273, 0x2252, 0x52b5, 0x4294, 0x72f7, 0x62d6,
    0x9339, 0x8318, 0xb37b, 0xa35a, 0xd3bd, 0xc39c, 0xf3ff, 0xe3de,
    0x2462, 0x3443, 0x0420, 0x1401, 0x64e6, 0x74c7, 0x44a4, 0x5485,
    0xa56a, 0xb54b, 0x8528, 0x9509, 0xe5ee, 0xf5cf, 0xc5ac, 0xd58d,
    0x3653, 0x2672, 0x1611, 0x0630, 0x76d7, 0x66f6, 0x5695, 0x46b4,
    0xb75b, 0xa77a, 0x9719, 0x8738, 0xf7df, 0xe7fe, 0xd79d, 0xc7bc,
    0x48c4, 0x58e5, 0x6886, 0x78a7, 0x0840, 0x1861, 0x2802, 0x3823,
    0xc9cc, 0xd9ed, 0xe98e, 0xf9af, 0x8948, 0x9969, 0xa90a, 0xb92b,
    0x5af5, 0x4ad4, 0x7ab7, 0x6a96, 0x1a71, 0x0a50, 0x3a33, 0x2a12,
    0xdbfd, 0xcbdc, 0xfbbf, 0xeb9e, 0x9b79, 0x8b58, 0xbb3b, 0xab1a,
    0x6ca6, 0x7c87, 0x4ce4, 0x5cc5, 0x2c22, 0x3c03, 0x0c60, 0x1c41,
    0xedae, 0xfd8f, 0xcdec, 0xddcd, 0xad2a, 0xbd0b, 0x8d68, 0x9d49,
    0x7e97, 0x6eb6, 0x5ed5, 0x4ef4, 0x3e13, 0x2e32, 0x1e51, 0x0e70,
    0xff9f, 0xefbe, 0xdfdd, 0xcffc, 0xbf1b, 0xaf3a, 0x9f59, 0x8f78,
    0x9188, 0x81a9, 0xb1ca, 0xa1eb, 0xd10c, 0xc12d, 0xf14e, 0xe16f,
    0x1080, 0x00a1, 0x30c2, 0x20e3, 0x5004, 0x4025, 0x7046, 0x6067,
    0x83b9, 0x9398, 0xa3fb, 0xb3da, 0xc33d, 0xd31c, 0xe37f, 0xf35e,
    0x02b1, 0x1290, 0x22f3, 0x32d2, 0x4235, 0x5214, 0x6277, 0x7256,
    0xb5ea, 0xa5cb, 0x95a8, 0x8589, 0xf56e, 0xe54f, 0xd52c, 0xc50d,
    0x34e2, 0x24c3, 0x14a0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
    0xa7db, 0xb7fa, 0x8799, 0x97b8, 0xe75f, 0xf77e, 0xc71d, 0xd73c,
    0x26d3, 0x36f2, 0x0691, 0x16b0, 0x6657, 0x7676, 0x4615, 0x5634,
    0xd94c, 0xc96d, 0xf90e, 0xe92f, 0x99c8, 0x89e9, 0xb98a, 0xa9ab,
    0x5844, 0x4865, 0x7806, 0x6827, 0x18c0, 0x08e1, 0x3882, 0x28a3,
    0xcb7d, 0xdb5c, 0xeb3f, 0xfb1e, 0x8bf9, 0x9bd8, 0xabbb, 0xbb9a,
    0x4a75, 0x5a54, 0x6a37, 0x7a16, 0x0af1, 0x1ad0, 0x2ab3, 0x3a92,
    0xfd2e, 0xed0f, 0xdd6c, 0xcd4d, 0xbdaa, 0xad8b, 0x9de8, 0x8dc9,
    0x7c26, 0x6c07, 0x5c64, 0x4c45, 0x3ca2, 0x2c83, 0x1ce0, 0x0cc1,
    0xef1f, 0xff3e, 0xcf5d, 0xdf7c, 0xaf9b, 0xbfba, 0x8fd9, 0x9ff8,
    0x6e17, 0x7e36, 0x4e55, 0x5e74, 0x2e93, 0x3eb2, 0x0ed1, 0x1ef0]


def show_irimage_ratio(file, ratioW, ratioH, offsetX=0, offsetY=0):
    ''' show ir image '''
    pio.renderers.default = 'iframe'  # jupyter에 보이도록 함
    f = open(file, "rb")
    data = f.read(160 * 120 * 2)
    crc = f.read(2)
    f.close()

    bindata = array.array('H')  # unsinged short format
    bindata.frombytes(data)  # 문자열 배열을  변환

    irdata = np.array(bindata).reshape(120, 160)

    maxv = irdata.max()
    minv = irdata.min()
    gap = maxv - minv
    image = np.floor(((irdata - minv) / gap) * 255)  #contrast

    w = 160 * np.sqrt(ratioW)
    h = 120 * np.sqrt(ratioH)
    wo = (160 - w) / 2
    ho = (120 - h) / 2
    x = [wo, wo, wo + w, wo + w, wo]
    y = [ho, ho + h, ho + h, ho, ho]
    targetX = np.array(x) + offsetX
    targetY = np.array(y) + offsetY

    fig = px.imshow(image)
    fig.add_trace(go.Scatter(x=targetX, y=targetY, marker_color='black'))
    fig.show()


def show_irimage_coordination(file, x, y, w, h):
    ''' start (x, y), width, height - show ir image '''
    pio.renderers.default = 'iframe'  # jupyter에 보이도록 함
    f = open(file, "rb")
    data = f.read(160 * 120 * 2)
    crc = f.read(2)
    f.close()

    bindata = array.array('H')  # unsinged short format
    bindata.frombytes(data)  # 문자열 배열을  변환
    irdata = np.array(bindata).reshape(120, 160)

    maxv = irdata.max()
    minv = irdata.min()
    gap = maxv - minv
    image = np.floor(((irdata - minv) / gap) * 255)  #contrast

    X = [x, x, x + w, x + w, x]
    Y = [y, y + h, y + h, y, y]
    fig = px.imshow(image)
    fig.add_trace(go.Scatter(x=X, y=Y, marker_color='black'))
    fig.show()


def crc_xmodem(bytedata):
    crc = 0
    datalen = len(bytedata)

    for i in range(datalen):
        crc = crc ^ (bytedata[i] << 8)
        crc = crc_table[crc >> 8] ^ (crc & 0xff)
    return crc


def load_irimage(filename):
    with open(filename, 'rb') as f:
        bindata = f.read(160 * 120 * 2)
        bincrc = f.read(2)

    calccrc = crc_xmodem(bindata)
    crc = bincrc[0] | (bincrc[1] << 8)
    if not calccrc == crc:
        print('crc error {}:{}'.format(calccrc, crc))
        raise ValueError("CRC error")

    irdata = array.array('H')
    irdata.frombytes(bindata)
    return irdata

    listdata = []
    k = 0
    for y in range(120):
        linedata = []
        for x in range(160):
            linedata.append(irdata[k])
            k = k + 1
        listdata.append(linedata)

    return listdata


def get_mean_temperature_target_ratio(filename, ratioW, ratioH, offsetX=0, offsetY=0):
    ''' 대상 크기를 조정하고, 위치를 이동할 수 있게함'''
    irdata = np.array(load_irimage(filename)).reshape(120, 160)
    temp_data = (irdata - 27315) / 100

    print('org: ', temp_data.min(), temp_data.max(), temp_data.shape, round(temp_data.mean(), 2))

    ''' target 위치 설정'''
    width = 160
    height = 120
    target_w = int(width * np.sqrt(ratioW))
    target_h = int(height * np.sqrt(ratioH))
    offset_w = int((width - target_w) / 2) + offsetX  # 타켓의 위치 이동
    offset_h = int((height - target_h) / 2) + offsetY

    target = temp_data[offset_w:(offset_w + target_w), offset_h:(offset_h + target_h)]
    avg_temp = round(target.mean(), 1)

    print('target: ', target.min(), target.max(), target.shape, round(target.mean(), 2))
    return target.min(), target.max(), avg_temp


def get_mean_temperature_coordination(filename, x, y, w, h):
    ''' 대상 크기를 조정하고, 위치를 이동할 수 있게함'''
    irdata = np.array(load_irimage(filename)).reshape(120, 160)
    temp_data = (irdata - 27315) / 100

    print('org: ', temp_data.min(), temp_data.max(), temp_data.shape, round(temp_data.mean(), 2))

    ''' target 위치 설정'''
    width = 160
    height = 120
    target = temp_data[x:x + w, y:y + h]
    avg_temp = round(target.mean(), 1)

    print('target: ', target.min(), target.max(), target.shape, round(target.mean(), 2))
    return target.min(), target.max(), avg_temp


def set_calibration_temp(value):
    pass


#if __name__ == '__main__':
def showIRImages_ratio(file, w_ratio=0.1, h_ratio=0.1, w_offset=0, h_offset=0):
    for f in file:
        min_temp, max_temp, mean_temp = get_mean_temperature_target_ratio(f, w_ratio, h_ratio, w_offset, h_offset)
        show_irimage_ratio(f, w_ratio, h_ratio, w_offset, h_offset)


def showIRImages_coordination(file, x=40, y=10, w=45, h=20):
    for f in file:
        show_irimage_coordination(f, x, y, w, h)
        get_mean_temperature_coordination(f, x, y, w, h)


def showRGBImages(file):
    for f in file:
        image = plt.imread(f)
        plt.imshow(image)
        plt.show()
        return image

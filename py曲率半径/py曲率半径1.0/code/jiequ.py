import matplotlib.pyplot as plt
import numpy as np

def flybhxqyjgb (fshift,xxdx,gszq):
### 傅里叶变换选取一级光斑
### fshift：傅里叶变换后二维数组
### xwdx：图像像素对应的实际大小
### gszq：光栅周期，如果旋转45度周期要除以根号2
    # xxdx = 0.65e-6
    # gszq = p1 / np.sqrt(2)
    julix = len(fshift[0]) * xxdx / gszq
    julix=146
    julix2 = int(julix / 2)
    juliy = len(fshift) * xxdx / gszq
    juliy=146
    juliy2 = int(juliy / 2)
    print("傅里叶变换选取一级光斑")
    print("计算一阶与0阶的x和y距离",julix, juliy)
    abs_fshift = np.abs(fshift)
    centrex = int(len(abs_fshift[0]) / 2)
    centrey = int(len(abs_fshift) / 2)
    print("计算的中心点",centrex, centrey)
    ####傅里叶变换后（1,0）位置截取的片段
    jqpd10 = abs_fshift[(centrey - juliy2):(centrey + juliy2), (centrex + julix2):(centrex + julix2 * 3)]
    max10 = np.max(jqpd10)
    w1 = np.where(jqpd10 == max10)
    wz10x = w1[1][0] + centrex + julix2
    wz10y = w1[0][0] + centrey - juliy2
    print("一阶频谱（1,0）点位置是：", wz10x, wz10y,'大小是：',max10)
    ####傅里叶变换后（0,1）位置截取的片段
    jqpd01 = abs_fshift[(centrey + juliy2):(centrey + juliy2 * 3), (centrex - julix2):(centrex + julix2)]
    max01 = np.max(jqpd01)
    w2 = np.where(jqpd01 == max01)
    print(w2)
    wz01x = w2[1][0] + centrex - julix2
    wz01y = w2[0][0] + centrey + juliy2
    print("一阶频谱（0,1）点位置是：", wz01x, wz01y,'大小是：',max01)
    juliy2 = int((wz01y - centrey) / 2)
    julix2 = int((wz10x - centrex) / 2)
    print('julix2=', julix2, '   juliy2=', juliy2)
    ####频谱01和10的片段
    pp01 = fshift[(wz01y - juliy2):(wz01y + juliy2), (wz01x - julix2):(wz01x + julix2)]
    pp10 = fshift[(wz10y - juliy2):(wz10y + juliy2), (wz10x - julix2):(wz10x + julix2)]
    re = [pp01, pp10]
    return re


def flybhsdxqyjgb (fshift,central_place,distance):
### 傅里叶变换手动选取一级光斑
### fshift：傅里叶变换后二维数组
### central_place：中心位置
### distance：中心到一阶中心的距离的一半
    pp01 = fshift[(central_place - distance):(central_place + distance), (central_place + distance):(central_place + distance*3)]
    pp10 = fshift[(central_place + distance):(central_place + distance*3), (central_place - distance):(central_place + distance)]
    # re = [pp01, pp10]
    return [pp01, pp10]
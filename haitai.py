import time
import numpy as np
import serial

class HaiTai:
    def __init__(self, device='COM3'):
        self.ser = None
        self.device =device

    def connect(self):
        self.ser = serial.Serial(self.device,115200,timeout=0)
        print("串口详情参数：", self.ser)
        if(self.ser.isOpen() == False):
            self.ser.open()
            # self.enter_motor_mode()
            return True
        return False

    def close(self):
        self.ser.close()

    def calc_crc16(self, data):
        putdata = b''
        for i in range(0,len(data)):
            putdata = putdata + self.num2str(data[i])
        # data = bytearray.fromhex(putdata)
        data = putdata
        crc = 0xFFFF
        for pos in data:
                crc ^= pos
                for i in range(8):
                        if((crc & 1) != 0):
                                crc >>= 1
                                crc ^= 0xA001
                        else:
                                crc >>= 1
        res =  ((crc & 0xff) << 8) + (crc >> 8)
        # res = hex(res)
        res = "%04x"%res
        data1 =   bytes(res[0:2],'UTF-8') 
        data2 =     bytes(res[2:],'UTF-8')
        data1 = res[0:2]
        data2 = res[2:]
        # print( res,data1,data2)
        return [data1, data2]

    def float_to_uint(self,x, x_min, x_max, bits):
    	# /// Converts a float to an unsigned int, given range and number of bits ///
        x_max = np.float16(x_max)
        x_min = np.float16(x_min)
        x = np.float16(x)
        span = x_max - x_min
        offset = x_min
        res = np.uint16((x - offset) * ((float)((1 << bits) - 1)) / span)
        # print("===res.bit_length:",res)
        # print(res)
        return res

    def uint_to_float(self, x_int, x_min, x_max, bits):
        # /// converts unsigned int to float, given range and number of bits ///
        span = x_max - x_min
        offset = x_min
        # return (np.float16(x_int))*span/(np.float16((1<<bits)-1)) + offset
        return (x_int)*span/((1<<bits)-1) + offset

    def set_abs_angle(self,  angle=10,wait = True): #相对原点位置旋转，单位度, 最大2圈
        count = int(angle*16384.0/360)
        # print("===count:",count)
        # count = 1000
        count_str = "%04x"%count
        count_str_1 =  int(count_str[0:2],16)  
        count_str_2 =  int(count_str[2:],16)  
        # print("==count_str:",count_str,count_str_1,count_str_2)
        data = [0]*11
        data[0] = 0x3E  #协议头
        data[1] = 0x00  #包序号
        data[2] = 0x01  #设备地址
        data[3] = 0x55  #命令码
        data[4] = 0x04  #数据长度
        # data[5] = 0xE8  #数据字段 低位
        # data[6] = 0x03  #数据字段 高位  0x03E8=1000
        data[5] = count_str_2     #数据字段 最低位
        data[6] = count_str_1     #数据字段 低位 
        data[7] = 0x00  #数据字段 高位
        data[8] = 0x00  #数据字段 最高位 
        
        [data1, data2] = self.calc_crc16(data[0:9])
        # print("calc_crc16 res:",data1, data2)
        data[9] = int(data1,16)  #CRC16校验
        data[10] = int(data2,16) #【协议头】至【数据字段】字节进行 CRC16_MODBUS 校验;
        self.sent_to_arm(data)
        if wait:
            while 1:
                curr_angle = self.get_curr_angle()
                if  abs(angle-curr_angle )<0.05:
                    break
        return

    def reset(self):# 回到初始0位
        data = [0]*7
        data[0] = 0x3E  #协议头
        data[1] = 0x00  #包序号
        data[2] = 0x01  #设备地址
        data[3] = 0x51  #命令码
        data[4] = 0x00  #数据长度
        [data1, data2] = self.calc_crc16(data[0:5])
        data[5] =  int(data1,16)   #CRC16校验
        data[6] =  int(data2,16)  #【协议头】至【数据字段】字节进行 CRC16_MODBUS 校验;
        self.sent_to_arm(data)
        return

    def get_curr_angle(self): # 获取当前偏离0位角度
        data = [0]*7
        data[0] = 0x3E  #协议头
        data[1] = 0x00  #包序号
        data[2] = 0x01  #设备地址
        data[3] = 0x2F  #命令码
        data[4] = 0x00  #数据长度
        [data1, data2] = self.calc_crc16(data[0:5])
        data[5] =  int(data1,16)   #CRC16校验
        data[6] =  int(data2,16)  #【协议头】至【数据字段】字节进行 CRC16_MODBUS 校验;
        self.sent_to_arm(data)
        while 1: #
            recv = self.ser.read(1)
            # print("recv",recv)
            recv_int = int.from_bytes(recv,byteorder='little',signed=False)
            if recv_int == 0x3C:
                recvs = self.ser.read(15)
                # print("==recvs :",recvs )
                break
        recv_data = [0]*15
        recv_data[0] = recv
        recv_data[1:] = recvs
        # print("recv_data:",recv_data)
        angle_count = recv_data[7:11]
        angle_count = int.from_bytes(angle_count,byteorder='little',signed=True)
        # print(len(recv_data),angle_count)
        angle = angle_count*360.0/16384
        # print("当前角度:",angle)
        return angle 

    #把十六进制或十进制的数转成bytes
    def num2str(self,num):
        # print("num:",num)
        strdata = hex(num)
        strdata = strdata[2:]
        # print("strdata:",strdata)
        if(len(strdata) == 1):
            strdata = '0'+ strdata
        strdata = bytes.fromhex(strdata)     
        # print("strdata:",strdata,type(strdata))
        return strdata

    def sent_to_arm(self,data):
        def print_hex(bytes):
            res = " "
            l = [hex(int(i)) for i in bytes]
            res = res.join(l)
            return res
        putdata = b''
        for i in range(0,len(data)):
            putdata = putdata + self.num2str(data[i])
        # print('发送的数据：',print_hex(putdata))
        self.ser.write(putdata)
        return 
 
if __name__ == "__main__":
    haitai = HaiTai()
    haitai.connect()
    haitai.set_abs_angle(0,wait = True)
    time.sleep(1)
    for i in range(5):
        haitai.set_abs_angle(360,wait = True)  # 转到绝对360度位置        
        haitai.set_abs_angle(30,wait = True)
        angle = haitai.get_curr_angle()
        print("current angle:", angle)
    haitai.set_abs_angle(0,wait = True)
    haitai.close()

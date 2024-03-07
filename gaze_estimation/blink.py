from scipy.spatial import distance as dist
import numpy as np
import argparse
import time
import cv2

class Blink_Detector(object):
	def __init__(self,v_t=0.04,t_t=0.4):
		self.v_thres = 0.06
		self.t_thres = 0.6
		self.pre_time = 0
		self.previous_time=0
		self.blink_time = 0
		self.pre_ear = 0
		total=0
		self.flag_blinkonce=0
		self.flag_print=1
		#识别视线落点的阈值
		self.thres1=0.22
		#识别双眨眼触发时的闭眼阈值
		self.thres2=0.19
		self.counter=0
		self.double_blink=0
		#当第一次检测到人眼的ear时，设置self.thres1，self.thres2；把该值设为1
		self.start=0


	def eye_aspect_ratio(self, eye):
		# compute the euclidean distances between the two sets of
		# vertical eye landmarks (x, y)-coordinates
		A = dist.euclidean(eye[1], eye[5])
		B = dist.euclidean(eye[2], eye[4])

		# compute the euclidean distance between the horizontal
		# eye landmark (x, y)-coordinates
		C = dist.euclidean(eye[0], eye[3])

		# compute the eye aspect ratio
		ear = (A + B) / (2.0 * C)

		# return the eye aspect ratio
		return ear

	def detect(self,left,right):
		leftEAR = self.eye_aspect_ratio(left)
		rightEAR = self.eye_aspect_ratio(right)
		now_time = time.time()
		interval = now_time-self.pre_time
		self.pre_time = now_time
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
		diff_ear = abs((ear-self.pre_ear))/interval
		self.pre_ear = ear
		#print("ear:",ear)
		#根据检测到的人眼设置初始阈值
		if self.start == 0:
			self.start=1
			self.thres1=ear
			self.thres2=ear-0.4
		
		#判断是否可以识别视线落点
		if ear>=self.thres1:
			self.flag_print=1
		else:
			self.flag_print=0
		#判断是否触发双眨眼
		if ear>=self.thres2:
			if self.counter>=1:
				self.flag_blinkonce=1
				self.counter=0
				# 如果此时ear大于thres2(0.15)：
				# 那么如果之前小于thres_2的闭眼计数大于等于1，就认为此时是眨眼后的张开阶段，完成了一次眨眼。
				# 将self.flag_blinkonce计为1
			else:
				self.counter=0
				self.flag_blinkonce=0
				#否则，就是正常的睁眼阶段，把self.counter=0、self.flag_blinkonce=0都归0
		else:
			self.counter+=1
			self.flag_blinkonce=0
			# 当ear小于thres2时，是闭眼阶段，给self.counter+1，闭眼阶段认为没有完成眨眼，self.flag_blinkonce计为0
		#判断是否进行了双眨眼
		if self.flag_blinkonce==1:
			#当self.flag_blinkonce=1，即每一次成功眨眼时，首先计算此时的时间与上一次眨眼的时间间隔；
			interval=time.time()-self.previous_time
			#计算完成后，将最新一次的眨眼时间更新为此时的时间
			self.previous_time=time.time()
			#如果间隔满足 interval 的间距条件，也就是两次眨眼完成的时间差足够小，那么就认为双眨眼成功触发；
			# 反之双眨眼没有成功触发，self.double_blink=0
			if interval <=0.3:
				self.double_blink=1
				#print("interval:",interval)
				#print("success")
			else :
				self.double_blink=0

		#如果此时self.flag_blinkonce=0，那么一定不满足双眨眼的触发条件
		else:
			self.double_blink=0

		return self.double_blink,self.flag_print


		
		#徐浩洋学长根据 ear 变化率判断眨眼的方法
		# if diff_ear > self.v_thres:
		# 	if interval < 0.25:
		# 		self.blink_time += interval
				

		# # otherwise, the eye aspect ratio is not below the blink
		# # threshold
		# else:
		# 	if self.blink_time > self.t_thres:
		# 		print("close_eye")

		# 	if self.blink_time < self.t_thres and self.blink_time > 0.2:
		# 		self.blink_time = 0
		# 		return True
		# 	self.blink_time = 0
		# return False
	







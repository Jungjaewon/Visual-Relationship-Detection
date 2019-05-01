import caffe
import numpy as np
import json
import math
import os.path as ops
import os
import cv2
import h5py

import cv2
import json
import cPickle as cp
import sys
import os.path

class DataLayer(caffe.Layer):
	def _shuffle_inds(self):
		self._perm = np.random.permutation(np.arange(self._num_instance))
		self._cur = 0

	def _get_next_batch_ids(self):
		if self._cur + self._batch_size > self._num_instance:	
			self._shuffle_inds()
		ids = self._perm[self._cur : self._cur + self._batch_size]
		self._cur += self._batch_size	
		return ids

	def _getAppr(self, im, bb):
		subim = im[bb[1] : bb[3], bb[0] : bb[2], :]
		subim = cv2.resize(subim, None, None, 224.0 / subim.shape[1], 224.0 / subim.shape[0], interpolation=cv2.INTER_LINEAR)
		pixel_means = np.array([[[103.939, 116.779, 123.68]]])
		subim -= pixel_means
		subim = subim.transpose((2, 0, 1))
		return subim
		
	def _getUnionBBox(self, aBB, bBB, ih, iw, margin = 1):
		return [max(0, min(aBB[0], bBB[0]) - margin), \
			max(0, min(aBB[1], bBB[1]) - margin), \
			min(iw, max(aBB[2], bBB[2]) + margin), \
			min(ih, max(aBB[3], bBB[3]) + margin)]

	def _box_center(self,box):

    		w = float(box[2] - box[0])
    		h = float(box[3] - box[1])

    		return [box[0] + w/2, box[1] + h/2]
	def _box_size(self,box):

		w = float(box[2] - box[0])
		h = float(box[3] - box[1])

    		return w * h
	def _box_location(self,sb_box, ob_box, iw, ih):

		sb_location = self._box_center(sb_box)
		ob_location = self._box_center(ob_box)
		sb_location[0] = sb_location[0] / float(iw)
		sb_location[1] = sb_location[1] / float(ih)
		ob_location[0] = ob_location[0] / float(iw)
		ob_location[1] = ob_location[1] / float(ih)
		x = ob_location[0] - sb_location[0]
		y = ob_location[1] - sb_location[1]

	    	return x,y
	def _box_contain(self,a_box, b_box):

    		if a_box[0] < b_box[0] and b_box[2] < a_box[2] \
            		and a_box[1] < b_box[1] and b_box[3] < a_box[3]:
        		return True # 1
    		else:
        		return False # 0

	def _box_iou(self,sb_box, ob_box):
		x1 = max(sb_box[0],ob_box[0])
		y1 = max(sb_box[1],ob_box[1])
		x2 = min(sb_box[2],ob_box[2])
		y2 = min(sb_box[3],ob_box[3])

		interArea = (x2 - x1) * (y2 - y1)

		if interArea < 0:
			return 0.0

		sbBoxArea = (sb_box[2] - sb_box[0] ) * (sb_box[3] - sb_box[1] )
		obBoxArea = (ob_box[2] - ob_box[0] ) * (ob_box[3] - ob_box[1] )

		if float(sbBoxArea  + obBoxArea - interArea) <= 0.0:
			return 0.0

		iou = interArea / float(sbBoxArea  + obBoxArea - interArea)

		if iou < 0:
			return 0.0

		return iou

	def _getSpatial(self, sbj_bb, obj_bb, iw, ih): # This function returns proposed spatial vector.

		im_area = iw * ih

		s_xmin = sbj_bb[0]
		s_ymin = sbj_bb[1]
		s_xmax = sbj_bb[2]
		s_ymax = sbj_bb[3]
		sw = s_xmax - s_xmin
		sh = s_ymax - s_ymin
		sbj_area = sw * sh

		o_xmin = obj_bb[0]
		o_ymin = obj_bb[1]
		o_xmax = obj_bb[2]
		o_ymax = obj_bb[3]
		ow = o_xmax - o_xmin
		oh = o_ymax - o_ymin
		obj_area = ow * oh


		box_iou = self._box_iou(sbj_bb,obj_bb)
		lx, ly = self._box_location(sbj_bb, obj_bb, iw, ih)
		so_contain = self._box_contain(sbj_bb,obj_bb)
		os_contain = self._box_contain(obj_bb,sbj_bb)
		spatial_vector = [sbj_area/ float(im_area), obj_area / float(im_area), box_iou, lx, ly, so_contain,os_contain]

		return np.array(spatial_vector)

	def _getYuSpatial(self, sbj_bb, obj_bb, iw, ih): # This function returns spatial vector in Yu's paper for ablation experiments.

		im_area = iw * ih

		s_xmin = sbj_bb[0]
		s_ymin = sbj_bb[1]
		s_xmax = sbj_bb[2]
		s_ymax = sbj_bb[3]
		sw = s_xmax - s_xmin
		sh = s_ymax - s_ymin
		sbj_area = sw * sh

		o_xmin = obj_bb[0]
		o_ymin = obj_bb[1]
		o_xmax = obj_bb[2]
		o_ymax = obj_bb[3]
		ow = o_xmax - o_xmin
		oh = o_ymax - o_ymin
		obj_area = ow * oh


		spatial_vector = [s_xmin / float(sw), s_ymin / float(sh), s_xmax / float(sw), s_ymax / float(sh), sbj_area/ float(im_area),
				o_xmin / float(ow), o_ymin / float(oh), o_xmax / float(ow), o_ymax / float(oh), obj_area / float(im_area)]

		return np.array(spatial_vector)

	def load_wordvector(self, root_folder, filename): # This function loads word vectors in Lu's github.  each word vector is 300 dim.
	    w2v_dict = {}
	    with open(root_folder + filename,'r') as fp:
		w2v = fp
		vector_list = w2v.readlines()
		#print len(vector_list) 100
		for vector in vector_list:
		    splited = vector.split(' ')
		    len_obj = len(splited) -300 -1
		    obj = ''
		    for i in range(len_obj):
		        obj += splited[i] + ' '
		    obj = obj.strip()
		    #print obj
		    v_array = np.zeros(300)

		    for it in range(len_obj,len(splited) - 1):
		        v_array[it - len_obj] = np.float64(splited[it])
		        assert (it - len_obj) >= 0 and (it - len_obj) <= 299
		    w2v_dict[obj] = v_array

	    assert len(w2v_dict) == 100

	    return w2v_dict
	
	def _get_next_batch(self):
		ids = self._get_next_batch_ids()
		ims = []
		labels = []
		spatials = []
		pair_vectors = []
		for id in ids:
			sample = self._samples[id]

			if self.vg == 'VG':
				assert ops.isfile(self._root_path1 + sample["filename"]) or ops.isfile(self._root_path2 + sample["filename"])

				if ops.isfile(self._root_path1 + sample["filename"]):
					im = cv2.imread(self._root_path1 + sample["filename"]).astype(np.float32, copy=False)
				elif ops.isfile(self._root_path2 + sample["filename"]):
					im = cv2.imread(self._root_path2 + sample["filename"]).astype(np.float32, copy=False)
			else:
				im = cv2.imread(self._root_path + sample["filename"]).astype(np.float32, copy=False)
			ih = im.shape[0]	
			iw = im.shape[1]
			sbj_bbox = sample["subject"]
			obj_bbox = sample["object"]
			sbj_name = sample['phrase'][0]
			obj_name = sample['phrase'][2]
			'''
			labels.append(sample["label_phrase"][1])
			union_box = self._getUnionBBox(sample["subject"], sample["object"], ih, iw)			
			ims.append(self._getAppr(im, union_box))
			spatials.append(self. _getSpatial(sbj_bbox, obj_bbox, iw, ih))
			obj_pair = np.hstack((self._w2v_dict[sbj_name],self._w2v_dict[obj_name]))
			pair_vectors.append(obj_pair)
			'''
			if self._mode == 'L':
				obj_pair = np.hstack((self._w2v_dict[sbj_name],self._w2v_dict[obj_name]))
				pair_vectors.append(obj_pair)
				labels.append(sample["label_phrase"][1])
			elif self._mode == 'LS':
				obj_pair = np.hstack((self._w2v_dict[sbj_name],self._w2v_dict[obj_name]))
				if self.smode == 'Yu':
					spatials.append(self. _getYuSpatial(sbj_bbox, obj_bbox, iw, ih))
				else:
					spatials.append(self. _getSpatial(sbj_bbox, obj_bbox, iw, ih))
				pair_vectors.append(obj_pair)
				labels.append(sample["label_phrase"][1])
			elif self._mode == 'V':
				union_box = self._getUnionBBox(sample["subject"], sample["object"], ih, iw)			
				ims.append(self._getAppr(im, union_box))
				labels.append(sample["label_phrase"][1])
			elif self._mode in  ['L_V','L_VW','VW']:
				labels.append(sample["label_phrase"][1])
				union_box = self._getUnionBBox(sample["subject"], sample["object"], ih, iw)			
				ims.append(self._getAppr(im, union_box))
				obj_pair = np.hstack((self._w2v_dict[sbj_name],self._w2v_dict[obj_name]))
				pair_vectors.append(obj_pair)
			elif self._mode in ['SVW','L_SVW','LS_V','LS_VW','LS_SVW','LS_SV','L_SV','SV']:
				labels.append(sample["label_phrase"][1])
				union_box = self._getUnionBBox(sample["subject"], sample["object"], ih, iw)			
				ims.append(self._getAppr(im, union_box))
				if self.smode == 'Yu':
					spatials.append(self. _getYuSpatial(sbj_bbox, obj_bbox, iw, ih))
				else:
					spatials.append(self. _getSpatial(sbj_bbox, obj_bbox, iw, ih))
				obj_pair = np.hstack((self._w2v_dict[sbj_name],self._w2v_dict[obj_name]))
				pair_vectors.append(obj_pair)
				
		if self._mode == 'L':
			return_dict = {'labels' : np.array(labels), 'pair_vector': np.array(pair_vectors)}
		elif self._mode == 'LS':
			return_dict = {'labels' : np.array(labels), 'pair_vector': np.array(pair_vectors),'spatial' : np.array(spatials)}
		elif self._mode == 'V':
			return_dict = {'labels' : np.array(labels), 'ims' : np.array(ims)}
		elif self._mode in  ['L_V','L_VW','VW']:
			return_dict = {'labels' : np.array(labels),'ims' : np.array(ims),'pair_vector': np.array(pair_vectors)}
		elif self._mode in ['SVW','L_SVW','LS_V','LS_VW','LS_SVW','LS_SV','L_SV','SV']:
			return_dict = {'labels' : np.array(labels), 'ims' : np.array(ims),'pair_vector': np.array(pair_vectors),'spatial' : np.array(spatials)}

		return return_dict
	
	def setup(self, bottom, top):
		layer_params = json.loads(self.param_str)

		self._samples = json.load(open(layer_params["dataset"]))
		self._num_instance = len(self._samples)
		self._batch_size = layer_params["batch_size"]
		self._mode = layer_params["mode"] # which model to be trained ex) L, V, LS+V, LS+SVW
		if layer_params.has_key("smode"):
			self.smode = 'Yu'
		else:
			self.smode = None

		if layer_params.has_key("vg"):
			self.vg = 'VG'
		else:
			self.vg = None

		self._root_path = '/home/user01/py-faster-rcnn/data/vrd/Images/' # image path for vrd
		self._root_path1 = '/Data_ssd/VG_100K/' # image path for vg
		self._root_path2 = '/Data_ssd/VG_100K_2/' # image path for vg
		self._w2v_path = '/home/woodcook486/visual_relationships_experiment/word2vector/'
		self._w2v_dict = self.load_wordvector(self._w2v_path, 'word2vec_obj.txt')
		#self._name_to_top_map = {"ims": 0, "spatial" : 1, 'pair_vector' : 2, 'labels' : 3}
		if self._mode == 'L':
			self._name_to_top_map = {'pair_vector' : 0, 'labels' : 1}
			top[0].reshape(self._batch_size, 600)
			top[1].reshape(self._batch_size)
		elif self._mode == 'LS':
			self._name_to_top_map = {'spatial' : 0 ,'pair_vector' : 1, 'labels' : 2}
			top[0].reshape(self._batch_size, 7)			
			top[1].reshape(self._batch_size, 600)
			top[2].reshape(self._batch_size)
		elif self._mode == 'V':
			self._name_to_top_map = {'ims' : 0 , 'labels' : 1}
			top[0].reshape(self._batch_size, 3,224,224)			
			top[1].reshape(self._batch_size)
		elif self._mode in  ['L_V','L_VW','VW']:
			self._name_to_top_map = {"ims": 0, 'pair_vector' : 1, 'labels' : 2}
			top[0].reshape(self._batch_size, 3, 224, 224)
			top[1].reshape(self._batch_size, 600)
			top[2].reshape(self._batch_size)
		elif self._mode in ['SVW','L_SVW','LS_V','LS_VW','LS_SVW','LS_SV','L_SV','SV']:
			self._name_to_top_map = {"ims": 0, "spatial" : 1, 'pair_vector' : 2, 'labels' : 3}
			top[0].reshape(self._batch_size, 3, 224, 224)
			top[1].reshape(self._batch_size, 7)
			top[2].reshape(self._batch_size, 600)
			top[3].reshape(self._batch_size)

		if self.smode == 'Yu':
			top[self._name_to_top_map['spatial']].reshape(self._batch_size, 10)
		else:
			pass
			
		self._shuffle_inds()
		
	def forward(self, bottom, top):
		batch = self._get_next_batch()
		for blob_name, blob in batch.iteritems():
			idx = self._name_to_top_map[blob_name]
			top[idx].reshape(*(blob.shape))
			top[idx].data[...] = blob.astype(np.float32, copy=False)
	
	def backward(self, top, propagate_down, bottom):
		pass

	def reshape(self, bottom, top):
		pass


	

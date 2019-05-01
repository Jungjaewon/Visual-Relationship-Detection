import caffe
import argparse
import time, os, sys
import json
import cv2
import cPickle as cp
import numpy as np
import math
import os
import pprint
import mat4py
import os.path as ops


def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser()

	parser.add_argument('--gt_json_file', dest='gt_json_file', help='file containing image paths',default='', type=str)
    	parser.add_argument('--zgt_json_file', dest='zgt_json_file', help='file containing image paths',default='', type=str)
    	parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',default=0, type=int)
	parser.add_argument('--def', dest='prototxt',help='prototxt file defining the network',default='', type=str)
	parser.add_argument('--net', dest='caffemodel',help='model to test',default='', type=str)
	parser.add_argument('--prd_map', dest='prd_map',help='predicate mapping pickle file',default='', type=str)
	parser.add_argument('--dataset', dest='dataset',help='visaul relationship dataset',default='vrd', type=str)
	parser.add_argument('--maximum_det', dest='maximum_det', help='how many relationships te be detected in a image',default=50, type=int)
	parser.add_argument('--ov_thresh', dest='ov_thresh', help='score threshhold',default=0.5, type=float)
	parser.add_argument('--out_dir', dest='out_dir', help='score threshhold',default='', type=str)
	parser.add_argument('--mode', dest='mode', help='test mode',default='', type=str)
	parser.add_argument('--smode', dest='smode', help='test mode',default='N', type=str)
	# test_mode 0 only visual , 1 visual + spatial , 2 visual + wordvector , 3 visual + spatial + word vector
	# lang_module 0 do not use, 1 use

	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)

	args = parser.parse_args()
	return args

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def getDualMask(ih, iw, bb, size):
	rh = float(size) / ih
	rw = float(size) / iw
	x1 = max(0, int(math.floor(bb[0] * rw)))
	x2 = min(size, int(math.ceil(bb[2] * rw)))
	y1 = max(0, int(math.floor(bb[1] * rh)))
	y2 = min(size, int(math.ceil(bb[3] * rh)))
	mask = np.zeros((size, size))
	mask[y1 : y2, x1 : x2] = 1
	assert(mask.sum() == (y2 - y1) * (x2 - x1))
	return mask

def _getUnionBBox(aBB, bBB, ih, iw, margin = 1):
		return [max(0, min(aBB[0], bBB[0]) - margin), \
			max(0, min(aBB[1], bBB[1]) - margin), \
			min(iw, max(aBB[2], bBB[2]) + margin), \
			min(ih, max(aBB[3], bBB[3]) + margin)]

def _getAppr(im, bb):
		subim = im[bb[1] : bb[3], bb[0] : bb[2], :]
		subim = cv2.resize(subim, None, None, 224.0 / subim.shape[1], 224.0 / subim.shape[0], interpolation=cv2.INTER_LINEAR)
		pixel_means = np.array([[[103.939, 116.779, 123.68]]])
		subim -= pixel_means
		subim = subim.transpose((2, 0, 1))
		return subim
def _box_center(box):

    	w = float(box[2] - box[0])
    	h = float(box[3] - box[1])

    	return [box[0] + w/2, box[1] + h/2]
def _box_size(box):

	w = float(box[2] - box[0])
	h = float(box[3] - box[1])

  	return w * h
def _box_location(sb_box, ob_box, iw, ih):

	sb_location = _box_center(sb_box)
	ob_location = _box_center(ob_box)
	sb_location[0] = sb_location[0] / float(iw)
	sb_location[1] = sb_location[1] / float(ih)
	ob_location[0] = ob_location[0] / float(iw)
	ob_location[1] = ob_location[1] / float(ih)
	x = ob_location[0] - sb_location[0]
	y = ob_location[1] - sb_location[1]

    	return x,y
def _box_contain(a_box, b_box):

  	if a_box[0] < b_box[0] and b_box[2] < a_box[2] \
          	and a_box[1] < b_box[1] and b_box[3] < a_box[3]:
        	return True # 1
    	else:
       		return False # 0

def _box_iou(sb_box, ob_box):
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

def _getSpatial(sbj_bb,obj_bb, iw, ih):

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


	box_iou = _box_iou(sbj_bb,obj_bb)
	lx, ly = _box_location(sbj_bb, obj_bb, iw, ih)
	so_contain = _box_contain(sbj_bb,obj_bb)
	os_contain = _box_contain(obj_bb,sbj_bb)
	spatial_vector = [sbj_area/ float(im_area), obj_area / float(im_area), box_iou, lx, ly, so_contain,os_contain]

	return np.array(spatial_vector)

def _getYuSpatial(sbj_bb, obj_bb, iw, ih):

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

def compare_box(detBBs, gtBBs):
	assert len(detBBs) == 4
	assert len(gtBBs) == 4
	if detBBs[0] ==	gtBBs[0] and detBBs[1] == gtBBs[1] and detBBs[2] == gtBBs[2] and detBBs[3] == gtBBs[3]:
		return True
	else:
		return False

def computeArea(bb):
	return max(0, bb[2] - bb[0] + 1) * max(0, bb[3] - bb[1] + 1)

def computeIoU(bb1, bb2):
	ibb = [max(bb1[0], bb2[0]), \
		max(bb1[1], bb2[1]), \
		min(bb1[2], bb2[2]), \
		min(bb1[3], bb2[3])]
	iArea = computeArea(ibb)
	uArea = computeArea(bb1) + computeArea(bb2) - iArea
	return (iArea + 0.0) / uArea


def load_wordvector(root_folder, filename):
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


def load_Wb(root_folder, filename):
    with open(root_folder + filename, 'r') as fp:
        data = mat4py.loadmat(fp)
    W = np.array(data['W'])
    B = np.array(data['B'])

    return W,B

def forward_batch(net, pair_vector= None, spatial = None, ims = None, factor = None):

	forward_args = {}

	if pair_vector is not None:
		net.blobs["pair_vector"].reshape(*(pair_vector.shape))
		forward_args["pair_vector"] = pair_vector.astype(np.float32, copy=False)
	if spatial is not None:
		net.blobs["spatial"].reshape(*(spatial.shape))
		forward_args["spatial"] = spatial.astype(np.float32, copy=False)
	if ims is not None:
		net.blobs["ims"].reshape(*(ims.shape))
		forward_args["ims"] = ims.astype(np.float32, copy=False)
	assert factor is not None
	net.blobs["factor"].reshape(*(factor.shape))
	forward_args["factor"] = factor.astype(np.float32, copy=False)

	net_out = net.forward(**forward_args)
	itr_pred = net_out["pred"].copy()
	return itr_pred


def test_net(net, jsonfile, prd_map, out_dir, caffemodel, thresh, test_mode, smode):

	image_root1 = '/Data_ssd/VG_100K/'
	image_root2 = '/Data_ssd/VG_100K_2/'
	test_result_k1 = {}
	test_result_k70 = {}
	w2v_dict = load_wordvector('/home/woodcook486/visual_relationships_experiment/word2vector/', 'word2vec_obj.txt')
	W,B = load_Wb('/home/woodcook486/visual_relationships_experiment/word2vector/', 'Wb.mat')
	with open(jsonfile,'r') as fp:
		Rel = json.load(fp)
	with open(prd_map,'r') as fp:
		prd_mapping = cp.load(fp)
	iteration = caffemodel.split('/')[-1]
	#f_k1 = open(out_dir + iteration + '_k1_prd.txt','w')
	#f_k70 = open(out_dir + iteration + '_k70_prd.txt','w')

	inv_prd_map = {v: k for k, v in prd_mapping.iteritems()}

	check_prd_k1 = {}
    	check_prd_k70 = {}
	cnt = 0
	len_Rel = len(Rel)
	for rel in Rel:
		if (cnt + 1) % 100 == 0:
			print 'test : ', cnt + 1, '/', len_Rel
		filename = rel['filename']
		assert ops.isfile(image_root1 + filename) or ops.isfile(image_root2 + filename)

		if ops.isfile(image_root1 + filename):
			im = cv2.imread(image_root1 + filename).astype(np.float32, copy=False)
		elif ops.isfile(image_root2 + filename):
			im = cv2.imread(image_root2 + filename).astype(np.float32, copy=False)
		ih = im.shape[0]
		iw = im.shape[1]
		Relationships = rel['relationships']

		spo_list_k1  = []
		spo_list_k70 = []

		cnf_list_k1  = []
		cnf_list_k70 = []

		sobox_list_k1  = []
		sobox_list_k70 = []

		for relationship in Relationships:
			spatials = []
			ims = []
			pair_vectors = []
			factors = []
			sbj_name = relationship['phrase'][0]
			obj_name = relationship['phrase'][2]

			sbj_bbox = relationship['subject']
			obj_bbox = relationship['object']

			factor = 1.0
			np_factor = np.zeros(70)
			np_factor.fill(factor)
			factors.append(np_factor)

			if test_mode == 'L':
				obj_pair = np.hstack((w2v_dict[sbj_name],w2v_dict[obj_name]))
				pair_vectors.append(obj_pair)
				batch_result = forward_batch(net, np.array(pair_vectors), None, None, np.array(factors))[0]
			elif test_mode == 'LS':
				obj_pair = np.hstack((w2v_dict[sbj_name],w2v_dict[obj_name]))
				if smode == 'Y':
					spatials.append(_getYuSpatial(sbj_bbox, obj_bbox, iw, ih))
				else:
					spatials.append(_getSpatial(sbj_bbox, obj_bbox, iw, ih))
				pair_vectors.append(obj_pair)
				batch_result = forward_batch(net, np.array(pair_vectors), np.array(spatials), None, np.array(factors))[0]
			elif test_mode == 'V':
				union_box = _getUnionBBox(sbj_bbox, obj_bbox, ih, iw)			
				ims.append(_getAppr(im, union_box))
				batch_result = forward_batch(net, None, None, np.array(ims), np.array(factors))[0]
			elif test_mode in  ['L_V','L_VW','VW']:
				union_box = _getUnionBBox(sbj_bbox, obj_bbox, ih, iw)			
				ims.append(_getAppr(im, union_box))
				obj_pair = np.hstack((w2v_dict[sbj_name],w2v_dict[obj_name]))
				pair_vectors.append(obj_pair)
				batch_result = forward_batch(net, np.array(pair_vectors), None, np.array(ims), np.array(factors))[0]
			elif test_mode in ['SVW','L_SVW','LS_V','LS_VW','LS_SVW','L_SV','LS_SV','SV']:		
				union_box = _getUnionBBox(sbj_bbox, obj_bbox, ih, iw)			
				ims.append(_getAppr(im, union_box))
				if smode == 'Y':
					spatials.append(_getYuSpatial(sbj_bbox, obj_bbox, iw, ih))
				else:
					spatials.append(_getSpatial(sbj_bbox, obj_bbox, iw, ih))
				obj_pair = np.hstack((w2v_dict[sbj_name],w2v_dict[obj_name]))
				pair_vectors.append(obj_pair)
				batch_result = forward_batch(net, np.array(pair_vectors), np.array(spatials), np.array(ims), np.array(factors))[0]

			argmax = np.argmax(batch_result)
			choosed_k70 = np.argsort(batch_result)[::-1]


			spo_list_k1.append([sbj_name, prd_mapping[argmax],obj_name])
			cnf_list_k1.append([1,batch_result[argmax],1])
			sobox_list_k1.append([sbj_bbox,obj_bbox])

			if check_prd_k1.has_key(prd_mapping[argmax]):
				check_prd_k1[prd_mapping[argmax]] += 1
			else:
				check_prd_k1[prd_mapping[argmax]] = 1

			#prd_name = relationship['phrase'][1]
			#pstr_k1 = prd_mapping[argmax] + ' ' + "{0:.3f}".format(batch_result[argmax])
            		#pstr_k70 = ''

			
			for i in range(len(choosed_k70)):
				spo_list_k70.append([sbj_name, prd_mapping[choosed_k70[i]],obj_name])
				cnf_list_k70.append([1,batch_result[choosed_k70[i]],1])
				sobox_list_k70.append([sbj_bbox,obj_bbox])
                		#pstr_k70 += (prd_mapping[choosed_k70[i]] + ' ' + "{0:.3f}".format(batch_result[choosed_k70[i]]) + ' ')

            		#f_k1.write('{} / {} , relationships : {} {} {} {}\n'.format(str(cnt + 1), str(len_Rel), sbj_name, prd_name, obj_name, pstr_k1))
            		#f_k70.write('{} / {} , relationships : {} {} {} {}\n'.format(str(cnt + 1), str(len_Rel), sbj_name, prd_name, obj_name, pstr_k70))

		test_result_k1[filename] = [spo_list_k1,cnf_list_k1,sobox_list_k1]
		test_result_k70[filename] = [spo_list_k70,cnf_list_k70,sobox_list_k70]
		cnt += 1
	#f_k1.write(json.dumps(check_prd_k1,indent=2))
	#f_k1.close()
    	#f_k70.close()

	#print count_prd
	#weight = args.caffemodel.split('/')[-1]
	'''
	with open(out_dir + weight + '_predicate_recognition_result_kn.pickle','w') as fp:
		cp.dump(test_result_kn,fp)
	with open(out_dir + weight + '_predicate_recognition_result_k1.pickle','w') as fp:
		cp.dump(test_result_k1,fp)
	with open(out_dir + weight + '_predicate_recognition_result_k70.pickle','w') as fp:
		cp.dump(test_result_k70,fp)
	'''

	return test_result_k1,test_result_k70

def evalutaion(json_file, test_result_file, maximum_det, ov_thresh, k, caffemodel, out_dir, zeroshot = False):

    with open(json_file,'r') as fp:
        Rel = json.load(fp)

    assert k != ''

    tp = []
    fp = []
    score = []
    total_gts = 0
    #assert len(Rel) == len(test_result_file)
    cnt = 0
    len_Rel = len(Rel)
    for rel in Rel:
	if (cnt + 1) % 100 == 0:
		pass
		#print 'evaluation ', cnt + 1 , '/', len_Rel
	filename = rel['filename']
	gtRelationships = rel['relationships']
	num_gts = len(gtRelationships)
	total_gts += num_gts
	gt_detected = np.zeros(num_gts)

	assert filename in test_result_file
	assert len(test_result_file[filename]) == 3
	spo_list = test_result_file[filename][0]
	conf_list = test_result_file[filename][1]
	sobox_list = test_result_file[filename][2]
	assert len(spo_list) == len(conf_list) == len(sobox_list)
	conf_list = np.array(conf_list)
	spo_list = np.array(spo_list)
	sobox_list = np.array(sobox_list)

	if isinstance(conf_list, np.ndarray) and len(conf_list) > 0:
		det_score = np.log(conf_list[:,0] + 1e-8) + np.log(conf_list[:,1] + 1e-8) + np.log(conf_list[:,2] + 1e-8)
		inds = np.argsort(det_score)[::-1]

		if args.maximum_det > 0 and args.maximum_det < len(inds):
			inds = inds[:args.maximum_det]
		top_confs = conf_list[inds]
		top_spo = spo_list[inds]
		top_box = sobox_list[inds]
		top_score = det_score[inds]
		num_dets = len(inds)
		for j in xrange(num_dets):
			ov_max = 0
			arg_max = -1
			for s in xrange(num_gts):
				phrase = gtRelationships[s]['phrase']
				gt_sbjbox = gtRelationships[s]['subject']
				gt_objbox = gtRelationships[s]['object']
				boxflag = compare_box(top_box[j][0], gt_sbjbox) and compare_box(top_box[j][1], gt_objbox)
				#boxflag and
				#print top_spo[j][0], top_spo[j][1], top_spo[j][2],  phrase[0] , phrase[1] , phrase[2]
				if boxflag and gt_detected[s] == 0 and top_spo[j][0] == phrase[0] and top_spo[j][1] == phrase[1] and top_spo[j][2] == phrase[2]:
					arg_max = s
					#ov = top_score[j] * -1
		                	#if ov >= ov_thresh and ov > ov_max:
		                    	#	ov_max = ov
		                    	#	arg_max = s
			if arg_max != -1:
                    		gt_detected[arg_max] = 1
                    		tp.append(1)
                    		fp.append(0)
                	else:
                    		tp.append(0)
                    		fp.append(1)
                	score.append(top_score[j])
	cnt = cnt + 1
    score = np.array(score)
    tp = np.array(tp)
    fp = np.array(fp)
    inds = np.argsort(score)
    inds = inds[::-1]
    tp = tp[inds]
    fp = fp[inds]
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = (tp + 0.0) / total_gts
    top_recall = recall[-1]
    iteration = caffemodel.split('/')[-1]

    #sbj = np.zeros(num_of_obj)
    #sbj[relationship['label_phrase'][0] - 1] = 1
    if zeroshot is False:
        print 'predicate_dectection Recall@',str(maximum_det), ' k= ',str(k), ' : ', top_recall, ' ', str(iteration)
    elif zeroshot is True:
        print 'predicate_dectection Zeroshot Recall@',str(maximum_det), ' k= ',str(k), ' : ', top_recall, ' ', str(iteration)

    if zeroshot is False:
        result_txt = 'predicate_recall@' + str(maximum_det) + '_k'+str(k)+'.txt'
    elif zeroshot is True:
        result_txt = 'Zeroshot_predicate_recall@' + str(maximum_det) + '_k'+str(k)+'.txt'

    with open(out_dir + result_txt,'a') as fp:
        if zeroshot is False:
 	      fp.write('predicate_dectection Recall@' + str(maximum_det) +  ' k= ' + str(k) + ' : ' + str(top_recall) + ' ' + str(iteration) + '\n')
        elif zeroshot is True:
          fp.write('predicate_dectection Zeroshot Recall@' + str(maximum_det) +  ' k= ' + str(k) + ' : ' + str(top_recall) + ' ' + str(iteration) + '\n')

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    assert args.dataset in ['vrd','vg']
    if args.out_dir == '':
		raise Exception, 'please specify output directory'
    else:
		try:
				os.makedirs(args.out_dir)
		except OSError:
				if not os.path.isdir(args.out_dir):
					raise Exception, 'Can not create directory please check it up'
    if args.mode not in ['L','LS','V','VW','SVW','L_V','L_VW','L_SVW','LS_V','LS_VW','LS_SVW','L_SV','LS_SV','SV']:
		raise Exception, 'please check the test_mode'

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    test_result_k1, test_result_k70 = test_net(net, args.gt_json_file,args.prd_map,args.out_dir, args.caffemodel, 0.1, args.mode, args.smode )
    evalutaion(args.gt_json_file, test_result_k1, args.maximum_det, args.ov_thresh, '1', args.caffemodel, args.out_dir, zeroshot = False)
    evalutaion(args.gt_json_file, test_result_k70, args.maximum_det, args.ov_thresh, '70', args.caffemodel, args.out_dir, zeroshot = False)

    evalutaion(args.zgt_json_file, test_result_k1, args.maximum_det, args.ov_thresh, '1', args.caffemodel, args.out_dir, zeroshot = True)
    evalutaion(args.zgt_json_file, test_result_k70, args.maximum_det, args.ov_thresh, '70', args.caffemodel, args.out_dir, zeroshot = True)



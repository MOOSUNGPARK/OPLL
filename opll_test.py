import sys
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import os
import time
import cv2
import opll_test_config as cfg
from model6_deeplab_v3_plus_final import Model
import opll_test_patch_generator as pg
import cx_Oracle


os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU

class Test:
    def __init__(self, input_data_path, STUDYLST_ID, LEARNMODEL_ID, DTT_TST_ID):
        self.model = Model()
        self.opll_conn = cx_Oracle.connect(cfg.ORACLE_INFO)

        self.studylst_id = STUDYLST_ID
        self.learnmodel_id = LEARNMODEL_ID
        self.dtt_tst_id = DTT_TST_ID

        self.ckpt_path = './models/opll/' + str(self.learnmodel_id) + '/opll.ckpt'
        self.img_path = './imgs'

        self.input_path = input_data_path

        self.status = 'N'

    def test(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=50, var_list=tf.global_variables())
            saver.restore(sess, self.ckpt_path + '/opll.ckpt')

            whole_input, img_shape = pg.val_input_loader(self.input_path)

            positions = []

            val_X = pg.extract_patches_from_batch(imgs=whole_input, patch_shape=(cfg.PATCH_SIZE, cfg.PATCH_SIZE, 1), stride=cfg.VAL_STRIDE)
            print('val_X shape :', val_X.shape)
            print('>>> data preparing complete!')
            print_img_idx = 0

            pred_list = []
            for batch in tl.iterate.minibatches(inputs=val_X, targets=val_X, batch_size=cfg.BATCH_SIZE, shuffle=False):
                print_img_idx += 1
                batch_x, _ = batch
                val_feed_dict = {self.model.X: batch_x,
                                 self.model.training: False,
                                 self.model.drop_rate: 0}
                pred = sess.run([self.model.bg_pred], feed_dict=val_feed_dict)
                pred = np.array(pred)
                print('pred shape :', pred.shape)

                if cfg.PRED_THRES == None:
                    pass
                else:
                    pred = np.where(pred > cfg.PRED_THRES,1,0)

                print(np.max(pred))
                pred_list.append(np.squeeze(pred))
            print('pred_list len :', len(pred_list))
            preds = np.expand_dims(np.concatenate(pred_list, axis=0), axis=-1)
            print('preds shape :', preds.shape)

            recon_pred = pg.reconstruct_from_patches_nd(preds, [cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 3], cfg.VAL_STRIDE) * 255 # 255.0
            print('recon_pred shape:', recon_pred.shape)
            print('img_shape :', img_shape)
            recon_pred = cv2.resize(recon_pred, (img_shape[0], img_shape[1]), cv2.INTER_NEAREST)
            recon_pred = cv2.threshold(recon_pred, 254, 255, cv2.THRESH_BINARY)[1]

            k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

            recon_pred = cv2.dilate(recon_pred, k, iterations=1)
            recon_pred = cv2.erode(recon_pred, k, iterations=1)

            num_labels, markers, state, cent = cv2.connectedComponentsWithStats(recon_pred)

            if num_labels != 1:
                self.status = 'Y'
                for idx in range(1, num_labels):
                    x, y, w, h, size = state[idx]
                    infor_position = [w, h, x, y]
                    positions.append(infor_position)

            for imgs in positions:
                # print(imgs)
                width, height, x_left_up, y_left_up = imgs

                x_right_down = x_left_up + width
                y_right_down = y_left_up + height

                # {"start":{"x":170,"y":330},"end":{"x":210,"y":355}}
                handles = {"start": {"x": x_left_up, "y": y_left_up}, "end": {"x": x_right_down, "y": y_right_down}}

                cur_labelinfo = self.opll_conn.cursor()
                time.sleep(0.2)
                cur_labelinfo.execute("insert into DTT_TST_LABELRNG_INFO(TST_LABELRNG_ID, STUDYLST_ID, LEARNMODEL_ID, DTT_TST_ID, LABELRNG_CN_INFO, IMG_IDX_NO) "
                                      "values(DTT_TST_LABELRNG_INFO01_SEQ.nextval, :learnmodel_id, :dtt_tst_id, :labelrng_cn_info, :img_idx_no) "
                                       , {'learnmodel_id':self.learnmodel_id, 'dtt_tst_id':self.dtt_tst_id, 'labelrng_cn_info':str(handles), 'img_idx_no':str(0)})
                time.sleep(0.2)
                self.opll_conn.commit()
                time.sleep(0.2)
                cur_labelinfo.close()

            cur_illinfo = self.opll_conn.cursor()
            time.sleep(0.2)
            cur_illinfo.execute("update DTT_TST_INFO "
                                "set DTTRESLT_ILL_YN=:dttreslt_ill_yn "
                                "where LEARNMODEL_ID=:model_id and STUDYLST_ID=:studylst_id and DTT_TST_ID=:dtt_tst_id",
                                {'dttreslt_ill_yn':self.status, 'model_id': self.learnmodel_id, 'studylst_id': str(self.studylst_id), 'dtt_tst_id': str(self.dtt_tst_id)})
            self.opll_conn.commit()
            time.sleep(0.2)
            cur_illinfo.close()


if __name__ == "__main__":
    studylst_id = sys.argv[1]
    learnmodel_id = sys.argv[2]
    dtt_tst_id = sys.argv[3]

    study_uid = sys.argv[4]
    series_uid = sys.argv[5]

    data_root_path = os.path.join(cfg.RETRIEVE_PATH, 'data', 'test', str(study_uid), str(series_uid), 'input')
    data_dir = os.path.join(data_root_path, os.listdir(data_root_path)) # SOP uid
    data_path = os.path.join(data_dir, os.listdir(data_dir)) # File

    tester = Test(input_data_path=data_path, STUDYLST_ID=studylst_id, LEARNMODEL_ID=learnmodel_id, DTT_TST_ID=dtt_tst_id)
    tester.test()



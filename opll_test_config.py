
############### DATABASE ACCESS INFORMATION ##################

# ORACLE_INFO = 'hiraadmin/hira2018#@192.168.0.32:1521/orcl'
OID = 'hiraadmin'
OPW = 'hira2018#'
OIP = '192.168.0.32'
OPT = '1521'
OSID =  'orcl'
ORACLE_INFO = OID+'/'+OPW+'@'+OIP+':'+OPT+'/'+OSID


############### DATA RETRIEVE INFORMATION ##################

# "http://192.168.0.204:8080/dcm4chee-arc/aets/DCM4CHEE"
DSIP = '192.168.0.204'
DSPT = '8080'
DCM_SERVER_URL = "http://" + DSIP + ':' + DSPT + "/dcm4chee-arc/aets/DCM4CHEE"

RS_BASE_URL = DCM_SERVER_URL + "/rs"
RS_HEADERS = { 'Accept': "application/dicom+json",
               'cache-control': "no-cache"
               }

WADO_URL = DCM_SERVER_URL + "/wado"
RETRIEVE_PATH = "/home/bjh/workplace/retrieve_test"
RESPONSE_TIMEOUT = 10
LABEL_IP = '192.168.0.204'

###  File setting ###
DATA_FILE_TYPE = 'npy'
HM_THRESHOLD_TYPE = 'fuzzy'         # fuzzy, mean, median, valley
REBUILD_FULL_IMG = False      ############# Do Not Use except first time
REBUILD_PATCH_DATA = False          # train_mobile.py
SAVE_WHOLE_PATCHES = False
REBUILD_POSTURE_DATA = False
REBUILD_VAL_PATCH_DATA =False
RESTORE = False                      # load weights file
CKPT_DIR = '2018_10_12_11_24/'
SAVE_IMG = True
SAVE_RECON_IMG = False
IMG_SAVING_EPHOCH = 5
PATH_SLASH = '/' if MODE == 'linux' else '\\'
N_FILES = 5

### Data info ###
IMG_SIZE = [576, 480]
CV_VAL_IDX = 4                # 0 ~ 4
PATCH_SIZE = 288
HIST_MATCH = False
PATCH_WISE = True
AUGMENTATION = False
PATCH_STRIDE = 32
VAL_STRIDE = 32
PATCH_CUTLINE = 0.005
VAL_PATCH_RATIO = 0.025
PRED_THRES = 0.5

### IMG SETTING ###
TOP_RIGHT = 0.2
TOP_DOWN = 0.35
UPSIDE = 0.2
DOWNSIDE = 0.2
RIGHTSIDE = 0.2
LEFTSIDE = 0.1

N_PATCH_TO_IMG = (((IMG_SIZE[0] - PATCH_SIZE) // PATCH_STRIDE) + 1) * (((IMG_SIZE[1] - PATCH_SIZE) // PATCH_STRIDE) + 1)
N_VAL_PATCH_TO_IMG = (((IMG_SIZE[0] - PATCH_SIZE) // VAL_STRIDE) + 1) * (((IMG_SIZE[1] - PATCH_SIZE) // VAL_STRIDE) + 1)
print(N_PATCH_TO_IMG)
print(N_VAL_PATCH_TO_IMG)
USED_MODALITY = ['flair', 't1', 't1ce', 't2']
N_INPUT_CHANNEL = 1
LANDMARK_DIVIDE = 10
RGB_CHANNEL = 1
N_CLASS = 2
TRAIN_LABEL = [0, 1]
ET_LABEL = [0, 0, 0, 1]
TC_LABEL = [0, 1, 0, 1]
WT_LABEL = [0, 1, 1, 1]

### Common ###
EPOCHS = 1                      # epochs
SPLITS = 5                      # cross validation cnt
SAVING_EPOCH = 1                # save model/img every SAVING_EPOCH
BATCH_SIZE = 5
BUFFER_SIZE = 3000
INIT_N_FILTER = 36              # output n_channel(n_filter) of first conv layer
ACTIVATION_FUNC = 'elu'         # relu, lrelu, elu, prelu, selu
LOSS_FUNC = 'dice'              # dice, focal, cross_entropy, dice_sum, huber, weighted_cross_entropy
LAMBDA = [0.1, 0.9]   # weight of each loss [bg, fg]
OPTIMIZER = 'adam'           # adam, rmsprop, sgd
INIT_LEARNING_RATE = 3e-4
DECAY_RATE = 0.9
DECAY_STEP = 2000
DECAY_STAIRCASE = True
NORMALIZATION_TYPE = 'batch'    # batch, group, batch_match
N_LAYERS = [2, 2, 2]            # n_layers before each downsampling
N_LAYERS_HIGH = [1,1]       # n_high_layers before each downsampling
N_LAYERS_LOW = [1]          # n_low_layers before each downsampling
DEPTH = len(N_LAYERS)           # total downsampling cnt. if 4 then img size(192 -> 96 -> 48 -> 24 -> 12)
DEPTH_HIGH = len(N_LAYERS_HIGH)
DEPTH_LOW = len(N_LAYERS_LOW)
HM_THRESHOLD_TYPE = 'fuzzy'     # fuzzy, mean, median, valley, fuzzy_log
DOWNSAMPLING_TYPE = 'neighbor'  # neighbor, maxpool, avgpool
UPSAMPLING_TYPE = 'resize'         # resize, transpose, add, concat, avgpool
GROUP_N = 4                     # group size of group_conv & group_norm
INIT_DROPOUT_RATE = 0.15
DROPOUT_INCREASE_RATE = 1.3    # 1.11^10=2.8394

### Mobilenet ###
WIDTH_MULTIPLIER = 1.0          # out_channel = in_channel * width_multiplier

### Histogram Match ###
BATCH_MATCH_THRES = 'fuzzy'         # 'fuzzy', 'mean', 'median', 'valley', fuzzy_log
N_MATCH_DIVIDE = 10
STANDARD = False
SCALE = 1


### Resnet ###
RES_DEPTH = 3           # 3 for v2, 4 for v1
MODE = 'residual_v2'  # possible block : residual_block_v1, bottleneck_block_v1,residual_block_v2, bottleneck_block_v2
                        # possible mode :  bottleneck_v1_with_unet, bottleneck_v2_with_unet, residual_v1_with_unet, residual_v2_with_unet
                        # possible resnet : bottleneck_v1, residual_v1, bottleneck_v2, residual_v2
bottleneck = True
v2_depth = 92
choices = {
      18: [2, 2, 2, 2],        # recommends residual_block_v1
      34: [3, 4, 6, 3],        # recommends residual_block_v1
      50: [3, 4, 6, 3],        # bottleneck_block_v1
      101: [3, 4, 23, 3],     # bottleneck_block_v1
      152: [3, 8, 36, 3],     # bottleneck_block_v1
      200: [3, 24, 36, 3],    # bottleneck_block_v1
      108: [12, 12, 12],       # v2, n = (depth - 2) / 9, for bottleneck_block_v2
      164: [18, 18, 18],       # v2
      1001: [111, 111, 111],   # v2
      'another': [(v2_depth - 2) / 9, (v2_depth - 2) / 9,(v2_depth - 2) / 9]   #v2
  }
n_filter_chunks = choices[108]
# n_filter_chunks = [3,4,6,3]         # or assign numbers of channels you want
# n_filters = [16, 32, 64, 128]
INIT_FILTER = 8
kernel_size = 1
stride = [1,1]
n_blocks = 5
n_classes = 2
training = True
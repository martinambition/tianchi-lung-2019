#肺窗
LUNG_MIN_BOUND = -1000.0
LUNG_MAX_BOUND = 400.0

#纵膈窗
CHEST_MIN_BOUND = 40-350/2
CHEST_MAX_BOUND = 40+350/2

BINARY_THRESHOLD = -550

TRAIN_SEG_LEARNING_RATE = 1e-4
INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH, INPUT_CHANNEL, OUTPUT_CHANNEL = 64, 64, 64, 1, 1

#4个疾病+1个unknow
CLASSIFY_INPUT_WIDTH, CLASSIFY_INPUT_HEIGHT, CLASSIFY_INPUT_DEPTH, CLASSIFY_INPUT_CHANNEL,CLASSIFY_OUTPUT_CHANNEL \
    = 32, 32, 32, 1, 5

#路径
CT_PATH = '../dataset/*/*.mhd'
TEST_FOLDER='../dataset/testset'
TEST2_FOLDER='../dataset/testset2'
TEST2_PROCESS_PATH = './temp/test'

TRAIN_FOLDER='../dataset/trainset'
ANNOTATION_FILE = "../dataset/chestCT_round1_annotation.csv"
LOG_BASE_PATH = './output/training_logs'
SEG_LOG_DIR = LOG_BASE_PATH + '/seg-run-{}'
CLASSIFY_LOG_DIR = LOG_BASE_PATH + '/classify-run-{}'
PREPROCESS_PATH = './temp/preprocess'
PREPROCESS_PATH_LUNG= './temp/preprocess/lung'
PREPROCESS_PATH_MEIASTINAL= './temp/preprocess/mediastinal'
PREPROCESS_PATH_META = './temp/preprocess/meta'
PREPROCESS_GENERATOR_LUNG_PATH = './temp/generator/seg/lung'
PREPROCESS_GENERATOR_MEIASTINAL_PATH = './temp/generator/seg/mediastinal'

PREPROCESS_GENERATOR_CLASS_LUNG_PATH = './temp/generator/class/lung'
PREPROCESS_GENERATOR_CLASS_MEIASTINAL_PATH = './temp/generator/class/mediastinal'


label_dic = {1:u'结节', 5:u'索条',31:u'动脉硬化或钙化',32:u'淋巴结钙化'}
label_softmax= {1:1,5:2,31:3,32:4}
label_softmax_reverse = {0:0,1:1,2:5,3:31,4:32}

#分割正负样本比列, 1:3
#分类正负样本分割 1：1
TRAIN_SEG_POSITIVE_SAMPLE_RATIO = 0.6
TRAIN_CLASSIFY_POSITIVE_SAMPLE_RATIO = 0.5

#分割随机漂移范围
ENABLE_RANDOM_OFFSET = True
TRAIN_SEG_SAMPLE_RANDOM_OFFSET = 12
#分类随机漂移范围。分类的格子要小一半。
TRAIN_CLASSIFY_SAMPLE_RANDOM_OFFSET = 4

#Evaluate Frequency
TRAIN_SEG_EVALUATE_FREQ = 10

#Train param
TRAIN_EPOCHS = 100000000
TRAIN_EARLY_STOPPING = 10
TRAIN_BATCH_SIZE = 16
TRAIN_VALID_STEPS = 160
TRAIN_STEPS_PER_EPOCH = 1200


DEBUG_PLOT_WHEN_EVALUATING_SEG = False

# ResNet
RESNET_BLOCKS = 16
RESNET_SHRINKAGE_STEPS = 4
RESNET_INITIAL_FILTERS = 16
TRAIN_CLASSIFY_LEARNING_RATE = 1e-4

ZERO_CENTER = 0.25

#Pretrain weight
SEG_LUNG_TRAIN_WEIGHT= './output/training_logs/seg-run-lung-13-13/lung_checkpoint-04-0.6646.hdf5'
SEG_MEDIASTINAL_TRAIN_WEIGHT='/output/training_logs/seg-run-mediastinal-16-16/mediastinal_checkpoint-07-0.5245.hdf5'
CLASS_LUNG_TRAIN_WEIGHT='./output/training_logs/classify-run-lung-2-18/lung_checkpoint-01-2.3591.hdf5'
import os
import tree_modules
import loss
import kd_tree

TREE_DATASET_DIR = os.path.join(os.getcwd(), 'datasets')

# adjustable hyperparameters
pc_size = 2048
lvls =3
epochs = 1
capacity = 128
batch_size = 64

i_isect = 1.0
t_isect = 1.2
t = 1.0
learning_rate = 0.0001
gamma = 1.0
beta = 1.0
train_unbalanced = False
sah_frag_name = '_fragments_{0}_sah'.format(pc_size)
vh_frag_name = '_fragments_{0}_vh'.format(pc_size)

init_config = {
    'point_cloud_size' : pc_size,
    'intersection_cost' : i_isect,
    'traversal_cost' : t_isect,
    't' : t,
    'gamma' : gamma,
    'layer_gamma' : 4.0,
    'beta' : beta,
    'penalty_fn' : loss.penalty_tree_loss(slope=1.0),
    'loss_fn' : loss.unsupervised_tree_loss(),
    'p_fn' : tree_modules.p_eval(t_isect),
    'q_fn' : tree_modules.q_eval(i_isect, beta),
    'greedy_q_fn' : tree_modules.gr_q_eval(i_isect),
    'norm_factor' : 1.0 / (pc_size * i_isect),
    'tree_levels' : lvls,
    'dense_units_point_enc' : capacity,
    'dense_units_regr' : capacity,
    'learning_rate' : learning_rate,
    'train_unbalanced' : train_unbalanced,
    'checkpoint_window' : 15,
    'epochs' : epochs,
    'batch_size' : batch_size, }

def buildNetworkName(strat, lvls, pc_size, capacity) :
    return '{0}_kdtree_{1}lvl_{2}pc_{3}capacity'.format(
        'sah' if strat == kd_tree.strategy.SURFACE_HEURISTIC_GREEDY else 'vh',
        str(lvls),
        str(pc_size),
        str(capacity),)

sah_config = init_config.copy()

sah_config['tree_strat'] = kd_tree.strategy.SURFACE_HEURISTIC_GREEDY

sah_config['name'] = buildNetworkName(
    sah_config['tree_strat'], sah_config['tree_levels'],
    sah_config['point_cloud_size'],
    sah_config['dense_units_point_enc'])

sah_config['train_dir'] = os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'train' + sah_frag_name)
#sah_config['train_dir'] = os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'train_fragments_2048_masked_sah')
sah_config['test_dir'] = os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'test' + sah_frag_name)
sah_config['valid_dir'] = None#os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'valid' + sah_frag_name),

sah_config['train_csv'] = os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'train' + sah_frag_name + '.csv')
#sah_config['train_csv'] = os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'train_fragments_2048_masked_sah.csv')
sah_config['test_csv'] = os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'test' + sah_frag_name + '.csv')
sah_config['valid_csv'] = None#os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'valid' + sah_frag_name + '.csv')

sah_config['weight_fn'] = tree_modules.sah_eval()
sah_config['pooling_fn'] = tree_modules.pool_treelet(t, 4 if init_config['train_unbalanced'] else 3,
    tree_modules.p_eval(t_isect),
    tree_modules.q_eval(i_isect, beta),
    tree_modules.gr_q_eval(i_isect),
    tree_modules.sah_eval(),
    init_config['norm_factor'])
    
vh_config = init_config.copy()

vh_config['tree_strat'] = kd_tree.strategy.VOLUME_HEURISTIC_GREEDY

vh_config['name'] = buildNetworkName(
    vh_config['tree_strat'], vh_config['tree_levels'],
    vh_config['point_cloud_size'],
    vh_config['dense_units_point_enc'])

vh_config['train_dir'] = os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'train' + vh_frag_name)
vh_config['test_dir'] = os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'test' + vh_frag_name)
vh_config['valid_dir'] = None#os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'valid' + sah_frag_name),

vh_config['train_csv'] = os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'train' + vh_frag_name + '.csv')
vh_config['test_csv'] = os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'test' + vh_frag_name + '.csv')
vh_config['valid_csv'] = None#os.path.join(TREE_DATASET_DIR, 'custom_scenes', 'valid' + sah_frag_name + '.csv')

vh_config['weight_fn'] = tree_modules.vh_eval()
vh_config['pooling_fn'] = tree_modules.pool_treelet(t, 4 if init_config['train_unbalanced'] else 3,
    tree_modules.p_eval(t_isect),
    tree_modules.q_eval(i_isect, init_config['beta']),
    tree_modules.gr_q_eval(i_isect),
    tree_modules.vh_eval(),
    init_config['norm_factor'])

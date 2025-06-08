model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        graph_cfg=dict(layout='hand_body_27'),
        in_channels=3, # 2 channels for x, y
        tcn_type='mstcn',
    ),
    cls_head=dict(
        type='GCNHead',
        num_classes=27,
        in_channels=256,
    ),
)

dataset_type = 'PoseDataset'
left_kp = [8, 9, 10, 11, 12, 13, 14, 15, 16]
right_kp = [18, 19, 20, 21, 22, 23, 24, 25, 26]

train_pipeline = [
    dict(type='PreNormalize2D'),  
    dict(type='GenSkeFeat', dataset='hand_body_27', feats=['j']),
    dict(type='UniformSample', clip_len=100),  
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

val_pipeline = [
    dict(type='PreNormalize2D'),  
    dict(type='GenSkeFeat', dataset='hand_body_27', feats=['j']),
    dict(type='UniformSample', clip_len=100),  
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

test_pipeline = [
    dict(type='PreNormalize2D'),  
    dict(type='GenSkeFeat', dataset='hand_body_27', feats=['j']),
    dict(type='UniformSample', clip_len=100),  
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

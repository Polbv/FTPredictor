from mmpose.apis import MMPoseInferencer
import mmdet
import mmcv
import mmpose
import os
import utils

def inference(path,pred_path,vis_path):

    
    vid_path = path   
    pred_path='data/predictions/aux/json'
    vis_path='data/predictions/aux/video'

    # instantiate the inferencer using the model alias
    inferencer = MMPoseInferencer('human')
    
   
    result_generator = inferencer(vid_path, pred_out_dir=pred_path, vis_out_dir=vis_path)
    for r in result_generator:
        
        try: 
            result = next(result_generator)
        except: pass

    return result





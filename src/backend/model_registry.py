"""Load the verified PCB detector checkpoints."""
from pathlib import Path
import base64
import hashlib
import json
import os
from threading import RLock
import time

ROOT=Path(__file__).resolve().parents[2]
PROJECT_MODEL='pcb_defectdet_k100_r100_missing_hole_layout11'
PROJECT_CONFIDENCE=.325
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'data/runtime/ultralytics'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'data/runtime/matplotlib'))


class ModelRegistry:
    def __init__(self, model_types=None):
        self.folders={name:ROOT/'artifacts/models'/name
                      for name in ('source','target','k500','k500_coco','pcb50','pcb100','pcb100_1280',
                                   'pcb_defectdet_k50','pcb_defectdet_k100_r100',PROJECT_MODEL)}
        if model_types is not None:
            self.folders={name:self.folders[name] for name in model_types}
        self.default_model=next(iter(self.folders))
        self.loaded={};self.lock=RLock()

    def canonical(self,name):
        name='target' if name=='adapted' else name
        if name not in self.folders:
            raise ValueError('Available detector IDs: '+', '.join(self.folders))
        return name

    def list_models(self):
        records=[]
        for name,folder in self.folders.items():
            path=folder/'model_info.json';info=json.loads(path.read_text()) if path.is_file() else {}
            is_target=name=='target'
            records.append({'model_type':name,'model_name':'YOLOv8s supervised target' if is_target else 'YOLOv8s source',
                'available':bool(info) and (folder/'model.pt').is_file(),
                'checkpoint_sha256':info.get('checkpoint_sha256'),
                'classes':info.get('classes',[]),'evaluated_class_ids':info.get('evaluated_class_ids',list(range(6))),
                'map_score':info.get('development_test_mAP50') if is_target else info.get('baseline_test_mAP50'),
                'map50_95':info.get('development_test_mAP50_95') if is_target else info.get('baseline_test_mAP50_95'),
                'precision':info.get('development_test_precision') if is_target else None,
                'recall':info.get('development_test_recall') if is_target else None,
                'evaluation':'DsPCBSD+ development set, 992 images' if is_target else 'PKU source holdout, 93 images',
                'method':'larger-label supervised fine-tuning' if is_target else 'source-domain supervised training',
                'image_size':info.get('image_size'), 'loaded':name in self.loaded})
            if name in ('k500','k500_coco'):
                records[-1].update(model_name='YOLOv8s COCO → K500' if name=='k500_coco' else 'YOLOv8s K500 adapted candidate',
                    map_score=info.get('validation_mAP50'), map50_95=info.get('validation_mAP50_95'),
                    precision=info.get('operating_point',{}).get('precision'),
                    recall=info.get('operating_point',{}).get('recall'),
                    evaluation='Target validation, 403 images; settings selected on this set, not a final test',
                    method=('COCO-pretrained detector fine-tuned on 500 labelled target images' if name=='k500_coco' else 'PCB-source adaptation using 500 labelled target images'),
                    operating_point=info.get('operating_point'), limitations=info.get('limitations',[]),
                    nms_iou=info.get('inference',{}).get('iou'))
            if name in ('pcb50','pcb100','pcb100_1280','pcb_defectdet_k50','pcb_defectdet_k100_r100',PROJECT_MODEL):
                point=info.get('operating_point',{})
                training_boards, development_boards = (50,180) if name=='pcb50' else (100,130)
                records[-1].update(model_name=f'YOLO11s · {training_boards}-board adaptation',
                    map_score=info.get('development_mAP50'), map50_95=info.get('development_mAP50_95'),
                    precision=point.get('precision'), recall=point.get('recall'),
                    evaluation=f'Mendeley PCB-Defect development set, {development_boards} whole-board scans',
                    method=f'Lazarus checkpoint fine-tuned on {training_boards} labelled whole-board scans',
                    operating_point=info.get('operating_point'), limitations=info.get('limitations',[]),
                    nms_iou=info.get('inference',{}).get('iou'))
            if name=='pcb_defectdet_k50':
                records[-1].update(model_name='YOLO11s · DefectDet K50',
                    method='Single-layer detector adapted with 50 double-layer images and 50 single-layer training images',
                    target_evaluation=info.get('target_evaluation'))
            if name=='pcb_defectdet_k100_r100':
                records[-1].update(model_name='YOLO11s · Balanced 1280',
                    method='Single-layer detector adapted with 100 double-layer images and 100 single-layer training images',
                    target_evaluation=info.get('target_evaluation'))
            if name==PROJECT_MODEL:
                records[-1].update(model_name='YOLO11s · Six defect classes',
                    method='Balanced 1280 fine-tuned with labeled PKU missing-hole examples',
                    class_confidence=info.get('class_confidence',{}),
                    missing_hole_evaluation=info.get('missing_hole_evaluation'),
                    target_evaluation=info.get('target_evaluation'))
        return records

    def infer(self,image,model_type,confidence,annotate):
        import cv2
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(1)
        name=self.canonical(model_type)
        with self.lock:
            if name not in self.loaded:
                folder=self.folders[name];info=json.loads((folder/'model_info.json').read_text())
                weights=folder/'model.pt'
                if hashlib.sha256(weights.read_bytes()).hexdigest()!=info['checkpoint_sha256']:
                    raise ValueError('Checkpoint checksum mismatch')
                model=YOLO(str(weights))
                if [model.names[i] for i in range(len(model.names))]!=info.get('checkpoint_classes',info['classes']):
                    raise ValueError('Checkpoint class order mismatch')
                self.loaded[name]=(model,info)
            model,info=self.loaded[name]
            start=time.perf_counter()
            inference=info.get('inference',{})
            options={key:inference[key] for key in ('iou','rect','max_det','agnostic_nms','augment') if key in inference}
            output_ids={int(raw):app_id for raw,app_id in info.get('output_class_ids',{}).items()}
            horizontal_mirror=inference.get('horizontal_mirror',False)
            options.update(imgsz=info['image_size'],conf=min(confidence,.001) if horizontal_mirror else confidence,
                classes=list(output_ids) if output_ids else info.get('evaluated_class_ids'),
                device=os.getenv('PCB_DEVICE','cpu'),verbose=False)
            result=model.predict(source=image,**options)[0]
            if horizontal_mirror:
                from backend.mirror_merge import merge_mirror_boxes
                mirror_size=inference.get('mirror_image_size',options['imgsz'])
                mirror_options={**options,'imgsz':mirror_size}
                reflected=model.predict(source=cv2.flip(image,1),**mirror_options)[0]
                boxes=merge_mirror_boxes(result.boxes.data.cpu(),reflected.boxes.data.cpu(),image.shape[1],
                                         nms_iou=options['iou'],max_det=options['max_det'])
                if inference.get('missing_hole_recovery'):
                    from backend.mirror_merge import confirm_other_defects, recover_large_missing_holes
                    low_options={**options,'imgsz':640}
                    low=model.predict(source=image,**low_options)[0]
                    low_mirror=model.predict(source=cv2.flip(image,1),**low_options)[0]
                    low_boxes=merge_mirror_boxes(low.boxes.data.cpu(),low_mirror.boxes.data.cpu(),image.shape[1],
                                                 nms_iou=options['iou'],max_det=options['max_det'])
                    missing_id=next(raw for raw,app_id in output_ids.items() if info['classes'][app_id]=='missing_hole')
                    hole_cutoff=info['class_confidence']['missing_hole']
                    boxes=recover_large_missing_holes(boxes,low_boxes,missing_id,confidence,hole_cutoff,
                                                      image.shape[0]*image.shape[1])
                    boxes=confirm_other_defects(boxes,low_boxes,missing_id,max(confidence,hole_cutoff))
                result.update(boxes=boxes[boxes[:,4]>=confidence])
            class_confidence=info.get('class_confidence',{})
            if class_confidence:
                boxes=result.boxes.data
                keep=torch.ones(len(boxes),dtype=torch.bool,device=boxes.device)
                for raw_id,class_id in output_ids.items():
                    cutoff=class_confidence.get(info['classes'][class_id],confidence)
                    keep &= (boxes[:,5]!=raw_id) | (boxes[:,4]>=cutoff)
                result.update(boxes=boxes[keep])
            elapsed=(time.perf_counter()-start)*1000
            detections=[]
            for box in result.boxes:
                raw_id=int(box.cls.item())
                class_id=output_ids[raw_id] if output_ids else raw_id
                detections.append({'class_id':class_id,'class_name':info['classes'][class_id],
                    'confidence':float(box.conf.item()),'bbox':[float(v) for v in box.xyxy[0].cpu().tolist()]})
            annotated=None
            if annotate:
                if output_ids:
                    result.names={raw:info['classes'][app_id] for raw,app_id in output_ids.items()}
                ok,encoded=cv2.imencode('.jpg',result.plot())
                if not ok:raise ValueError('Could not encode annotated image')
                annotated='data:image/jpeg;base64,'+base64.b64encode(encoded).decode()
            response={'detections':detections,'inference_time_ms':round(elapsed,2),
                    'checkpoint_sha256':info['checkpoint_sha256'],'image_width':image.shape[1],
                    'image_height':image.shape[0],'confidence_cutoff':confidence,'annotated_image':annotated}
            if horizontal_mirror:
                response['inference_mode']=('horizontal_mirror' if mirror_size==options['imgsz']
                                            else f'horizontal_mirror_{mirror_size}')
                if inference.get('missing_hole_recovery'):
                    response['inference_mode']+='_missing_hole_640_confirmed'
            return response

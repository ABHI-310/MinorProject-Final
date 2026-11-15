from _bootstrap import ROOT
"""Real-model API and database integration check; does not train models."""
from pathlib import Path
import argparse,base64,json,sys,tempfile
import uuid
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fastapi.testclient import TestClient
from backend.main import create_app
from backend.model_registry import PROJECT_CONFIDENCE, PROJECT_MODEL
from sqlalchemy import func,select
from database.db import Inspection,Detection,VerifiedModel


def verify(url,report):
    app=create_app(url,image_store=Path(report).resolve().parent/'verification_images')
    equipment='CHECK_'+uuid.uuid4().hex[:12]
    image=ROOT/'examples/pcb_defect_002.jpg'
    if not image.is_file():raise RuntimeError('Verification sample is missing: '+str(image))
    with TestClient(app) as client:
        health=client.get('/api/health');assert health.status_code==200
        model_records=client.get('/api/models').json()['models']
        assert len(model_records)==1 and model_records[0]['available']
        model=model_records[0]
        assert model['model_type']==PROJECT_MODEL
        image_upload={'file':(image.name,image.read_bytes(),'image/jpeg')}
        response=client.post('/api/detect',data={'equipment_id':equipment},files=image_upload)
        assert response.status_code==200,response.text
        value=response.json()
        assert value['model_type']==model['model_type'] and value['checkpoint_sha256']==model['checkpoint_sha256']
        assert value['confidence_cutoff']==model['operating_point']['confidence']==PROJECT_CONFIDENCE
        assert value['rul']['health_score'] is None and value['rul']['status']=='exposure_required'
        pixels=cv2.imdecode(np.frombuffer(base64.b64decode(value['annotated_image'].split(',')[1]),dtype=np.uint8),cv2.IMREAD_COLOR)
        assert pixels.shape[:2]==(value['image_height'],value['image_width'])
        assert value['detections'] and all(0<=d['bbox'][0]<d['bbox'][2]<=value['image_width'] and
                   0<=d['bbox'][1]<d['bbox'][3]<=value['image_height'] and
                   d['class_id'] in model['evaluated_class_ids'] and model['classes'][d['class_id']]==d['class_name']
                   for d in value['detections'])
        assert client.get(value['image_url']).status_code==200
        assert client.get(value['annotated_image_url']).status_code==200
        assert client.post('/api/detect',data={'model_type':'source'},files=image_upload).status_code==422
        assert value['count']==5
        target_sample=ROOT/'examples/img62.png'
        target_equipment='CHECK_'+uuid.uuid4().hex[:12]
        target_response=client.post('/api/detect',data={'equipment_id':target_equipment,'boards_inspected':1},
                                    files={'file':(target_sample.name,target_sample.read_bytes(),'image/png')})
        assert target_response.status_code==200,target_response.text
        target=target_response.json()
        assert target['model_type']==PROJECT_MODEL and target['checkpoint_sha256']==value['checkpoint_sha256']
        assert target['confidence_cutoff']==PROJECT_CONFIDENCE and target['inference_mode']==value['inference_mode']
        target_classes=model['target_evaluation']['classes']
        assert sorted(d['class_name'] for d in target['detections'])==sorted([*target_classes,'spurious_copper'])
        assert client.get(target['annotated_image_url']).status_code==200
        pku_sample=ROOT/'data/raw/pku/PCB_DATASET/images/Missing_hole/05_missing_hole_01.jpg'
        pku_response=client.post('/api/detect',data={'equipment_id':target_equipment},
                                 files={'file':(pku_sample.name,pku_sample.read_bytes(),'image/jpeg')})
        assert pku_response.status_code==200,pku_response.text
        pku=pku_response.json()
        assert pku['checkpoint_sha256']==value['checkpoint_sha256']
        hole_alerts=[d for d in pku['detections'] if d['class_name']=='missing_hole']
        assert len(hole_alerts)==3 and all(d['class_id']==0 and d['confidence']>=.40 for d in hole_alerts)
        assert model['class_confidence']=={'missing_hole':.4}
        results={PROJECT_MODEL:{'checkpoint_sha256':value['checkpoint_sha256'],'count':value['count'],
                          'inference_time_ms':value['inference_time_ms'],'annotated_image_decoded':True,
                          'double_layer_count':target['count'],'double_layer_classes':target_classes,
                          'synthetic_pku_missing_hole_count':len(hole_alerts)}}
        event_id=str(uuid.uuid4())
        data={'equipment_id':equipment,'event_id':event_id,'boards_inspected':100,'detections':[]}
        assert client.post('/api/events/detections',json=data).status_code==200
        assert client.post('/api/events/detections',json=data).json()['replayed'] is True
        for _ in range(4):
            data['event_id']=str(uuid.uuid4());assert client.post('/api/events/detections',json=data).status_code==200
        metrics=client.get('/api/metrics',params={'equipment_id':equipment}).json()
        assert metrics['inspection_count']==6 and metrics['rul']['status']=='history_unavailable'
        assert metrics['rul']['synthetic'] and metrics['rul']['rul_steps'] is None
        isolated=client.get('/api/metrics',params={'equipment_id':'UNUSED_'+uuid.uuid4().hex[:8]}).json()
        assert isolated['inspection_count']==0 and isolated['rul'] is None
        with app.state.db.session() as session:
            assert session.scalar(select(func.count()).select_from(VerifiedModel))>=1
        bad=client.post('/api/detect',files={'file':('bad.png',b'not an image','image/png')});assert bad.status_code==400
        demo=client.get('/api/rul/synthetic/EQP_000?limit=3').json();assert demo['synthetic'] is True and len(demo['timeline'])==3
        assert client.post('/api/adapt').status_code==410
    output={'status':'passed','database':health.json()['database'],'real_model_checks':results,
        'checks':['Single detector metadata and default inference on single-layer and double-layer samples','Archived detector unavailable in project API',
                  'Annotated JPEG decoding and image bounds',
                  'Verified model metrics persisted','Inspection, detections and outlook stored atomically',
                  'Zero-detection inspections retained','Retry idempotency','Equipment history isolation',
                  'No RUL without declared exposure','History-based simulated outlook after sufficient exposure',
                  'Invalid upload rejection','Phase 3 synthetic timeline endpoint','No API-triggered retraining'],
        'training_started':False,'rul_is_simulated':True}
    Path(report).write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2));return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--database-url');parser.add_argument('--report',type=Path,default=ROOT/'reports/phase4/sqlite_check.json');args=parser.parse_args()
    args.report.parent.mkdir(parents=True,exist_ok=True)
    if args.database_url:verify(args.database_url,args.report)
    else:
        with tempfile.TemporaryDirectory(prefix='pcb-api-check-') as folder:verify(f'sqlite:///{folder}/check.db',args.report)

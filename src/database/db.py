"""Inspection history on SQLite or PostgreSQL; existing legacy tables are preserved."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from threading import RLock
import uuid

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, create_engine, select, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool
from rul_synthesis.ensemble_runtime import METHOD, RUL_CAP, inspection_features, load_ensemble

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_URL = f"sqlite:///{ROOT/'data/application/maintenance.db'}"


def utcnow(): return datetime.now(timezone.utc)
def iso(value): return value.replace(tzinfo=timezone.utc).isoformat() if value.tzinfo is None else value.isoformat()


class Base(DeclarativeBase): pass


# Legacy tables retained without destructive schema changes.
class Detection(Base):
    __tablename__ = 'detections'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    equipment_id: Mapped[str] = mapped_column(String(50), index=True)
    image_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    defect_class: Mapped[str] = mapped_column(String(50))
    bbox_coords: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class RULEstimate(Base):
    __tablename__ = 'rul_estimates'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    equipment_id: Mapped[str] = mapped_column(String(50), index=True)
    health_score: Mapped[float] = mapped_column(Float)
    rul_cycles: Mapped[int] = mapped_column(Integer)
    defect_count: Mapped[int] = mapped_column(Integer)
    synthetic: Mapped[bool] = mapped_column(Boolean, default=True)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class ModelMetrics(Base):
    __tablename__ = 'model_metrics'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(100))
    model_type: Mapped[str] = mapped_column(String(50))
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    map_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    inference_time_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Inspection(Base):
    __tablename__ = 'inspections'
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    equipment_id: Mapped[str] = mapped_column(String(50), index=True)
    request_hash: Mapped[str] = mapped_column(String(64))
    model_type: Mapped[str] = mapped_column(String(20))
    image_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    boards_inspected: Mapped[int | None] = mapped_column(Integer, nullable=True)
    defect_count: Mapped[int] = mapped_column(Integer)
    detections_json: Mapped[str] = mapped_column(Text)
    details_json: Mapped[str] = mapped_column(Text)
    rul_json: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class VerifiedModel(Base):
    __tablename__ = 'verified_models'
    checkpoint_sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_type: Mapped[str] = mapped_column(String(20))
    report_json: Mapped[str] = mapped_column(Text)
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EventConflict(ValueError): pass


def inspection_dict(item):
    return {'inspection_id':item.id, 'equipment_id':item.equipment_id, 'model_type':item.model_type,
            'boards_inspected':item.boards_inspected, 'count':item.defect_count,
            'detections':json.loads(item.detections_json), 'rul':json.loads(item.rul_json),
            'timestamp':iso(item.timestamp), **json.loads(item.details_json)}


def simulated_outlook(history, predictor):
    """Build one model step per compatible PCB image, in inspection order."""
    result = dict(health_score=None, rul_cycles=None, rul_boards=None, rul_steps=None,
                  unit='inspection steps', cap_steps=RUL_CAP, method=METHOD, synthetic=True,
                  status='exposure_required', note='Enter one board for each PCB image to estimate RUL.')
    if not history or history[-1].boards_inspected is None:
        return result
    if predictor is None:
        return dict(result, status='model_unavailable', note='The trained RUL model files are unavailable.')

    observed = []
    settings = None
    for item in history:
        details = json.loads(item.details_json)
        width, height = details.get('image_width'), details.get('image_height')
        if item.boards_inspected != 1 or not width or not height:
            return dict(result, status='history_unavailable',
                        note='This estimate requires a history of single-board images for this equipment.')
        current_settings = (item.model_type, details.get('checkpoint_sha256'),
                            details.get('confidence_cutoff'), details.get('inference_mode'))
        if settings is not None and settings != current_settings:
            return dict(result, status='settings_changed',
                        note='Use a separate equipment ID for a history with different detector settings.')
        settings = current_settings
        observed.append(inspection_features(json.loads(item.detections_json), width, height))
    prediction = predictor.predict(observed)[-1]
    return dict(result, status='simulated_estimate', rul_steps=round(float(prediction), 2),
                history_inspections=len(observed),
                note='Simulated estimate based on saved inspections for this equipment.')


class Database:
    def __init__(self, url=None):
        url = url or os.getenv('DATABASE_URL',DEFAULT_DATABASE_URL)
        if url.startswith('postgres://'):url='postgresql+psycopg://'+url[len('postgres://'):]
        elif url.startswith('postgresql://'):url='postgresql+psycopg://'+url[len('postgresql://'):]
        kwargs={}
        if url.startswith('sqlite'):
            (ROOT/'data/application').mkdir(parents=True,exist_ok=True)
            kwargs['connect_args']={'check_same_thread':False,'timeout':30}
            if url in ('sqlite://','sqlite:///:memory:'):kwargs['poolclass']=StaticPool
        self.engine=create_engine(url,pool_pre_ping=True,**kwargs)
        self.sessions=sessionmaker(self.engine,expire_on_commit=False)
        self.write_lock=RLock()
        self.rul_predictor = None

    def initialise_rul(self):
        self.rul_predictor = load_ensemble()

    def initialise(self):Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self):
        session=self.sessions()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:session.close()

    def register_models(self, records):
        with self.session() as session:
            for record in records:
                if record.get('checkpoint_sha256') and record['available']:
                    session.merge(VerifiedModel(checkpoint_sha256=record['checkpoint_sha256'],
                        model_type=record['model_type'],report_json=json.dumps(record)))

    def replay(self,event_id,request_hash):
        with self.session() as session:
            item=session.get(Inspection,str(event_id))
            if item is None:return None
            if item.request_hash!=request_hash:raise EventConflict('event_id already belongs to different inspection data')
            return {**inspection_dict(item),'replayed':True}

    def record_inspection(self,equipment_id,detections,model_type,boards_inspected,request_hash,
                          event_id=None,image_path=None,details=None):
        event_id=str(event_id or uuid.uuid4())
        with self.write_lock:
            prior=self.replay(event_id,request_hash)
            if prior:return prior
            try:
                with self.session() as session:
                    item=Inspection(id=event_id,equipment_id=equipment_id,request_hash=request_hash,
                        model_type=model_type,image_path=image_path,boards_inspected=boards_inspected,
                        defect_count=len(detections),detections_json=json.dumps(detections),
                        details_json=json.dumps(details or {}),rul_json='{}',timestamp=utcnow())
                    session.add(item);session.flush()
                    history = [item]
                    if boards_inspected == 1 and self.rul_predictor is not None:
                        history = session.scalars(select(Inspection).where(Inspection.equipment_id == equipment_id)
                            .order_by(Inspection.timestamp, Inspection.id)).all()
                    rul = simulated_outlook(history, self.rul_predictor)
                    rul['predicted_at']=iso(item.timestamp)
                    item.rul_json=json.dumps(rul)
                    session.add_all([Detection(equipment_id=equipment_id,image_path=image_path,
                        defect_class=d['class_name'],bbox_coords=json.dumps(d['bbox']),confidence=d['confidence'],
                        timestamp=item.timestamp) for d in detections])
                    response={**inspection_dict(item),'replayed':False}
                return response
            except IntegrityError:
                prior=self.replay(event_id,request_hash)
                if prior:return prior
                raise

    def latest_metrics(self,equipment_id,hours=24):
        since=utcnow()-timedelta(hours=hours)
        with self.session() as session:
            conditions=[Detection.equipment_id==equipment_id,Detection.timestamp>=since]
            counts=dict(session.execute(select(Detection.defect_class,func.count()).where(*conditions)
                                        .group_by(Detection.defect_class)).all())
            events=session.scalars(select(Inspection).where(Inspection.equipment_id==equipment_id,
                Inspection.timestamp>=since).order_by(Inspection.timestamp.desc(),Inspection.id.desc()).limit(100)).all()
            total=session.scalar(select(func.count()).select_from(Inspection).where(
                Inspection.equipment_id==equipment_id,Inspection.timestamp>=since))
        return {'equipment_id':equipment_id,'hours':hours,'detection_count':sum(counts.values()),
                'defect_counts':counts,'inspection_count':total,'rul':json.loads(events[0].rul_json) if events else None,
                'history':[inspection_dict(item) for item in reversed(events)]}

    def dashboard_metrics(self,equipment_id,days=30):
        """Aggregate the entire selected window; only the history list is capped."""
        since=utcnow().replace(hour=0,minute=0,second=0,microsecond=0)-timedelta(days=days-1)
        with self.session() as session:
            conditions=[Inspection.equipment_id==equipment_id,Inspection.timestamp>=since]
            total=session.scalar(select(func.count()).select_from(Inspection).where(*conditions))
            defective=session.scalar(select(func.count()).select_from(Inspection).where(*conditions,Inspection.defect_count>0))
            events=session.scalars(select(Inspection).where(*conditions).order_by(
                Inspection.timestamp.desc(),Inspection.id.desc()).limit(100)).all()
            counts={};daily={}
            rows=session.execute(select(Detection.timestamp,Detection.defect_class).where(
                Detection.equipment_id==equipment_id,Detection.timestamp>=since))
            for timestamp,name in rows:
                day=timestamp.date().isoformat()
                counts[name]=counts.get(name,0)+1
                daily.setdefault(day,{})[name]=daily.get(day,{}).get(name,0)+1
        return {'equipment_id':equipment_id,'days':days,'since':iso(since),
                'inspection_count':total,'defective_inspections':defective,
                'detection_count':sum(counts.values()),'defect_counts':counts,'daily_defects':daily,
                'critical_detection_count':sum(counts.get(name,0) for name in ('short','open_circuit')),
                'history':[inspection_dict(item) for item in events], 'history_limit':100,
                'rul':json.loads(events[0].rul_json) if events else None}

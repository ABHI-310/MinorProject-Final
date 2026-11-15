import React,{useState} from 'react';
import {api,usePipeline} from '../data/PipelineContext';
export default function ModelSettings(){
 const [reviewDemo,setReviewDemo]=useState(null);
 const {data}=usePipeline();const [timeline,setTimeline]=useState(null),[error,setError]=useState('');
 return <><h2>Detector and Maintenance</h2><p>Evaluation scores are reported by board type.</p>
 <div className="pipeline-models">{data?.models?.map(m=><section key={m.model_type} className="ui-card pipeline-panel"><h3>{m.model_name}</h3><p>{m.available?'Available':'Missing'} · {m.method}</p><p>mAP50: {m.map_score!=null?(100*m.map_score).toFixed(2)+'%':'Unknown'} · mAP50–95: {m.map50_95!=null?(100*m.map50_95).toFixed(2)+'%':'Unknown'}</p><p>{m.evaluation}</p><p>Evaluated classes: {m.evaluated_class_ids.map(i=>m.classes[i]).join(', ')}</p><p>Inference image size: {m.image_size}</p>{m.operating_point&&<p>Validation at confidence {m.operating_point.confidence}: precision {(100*m.operating_point.precision).toFixed(2)}% · recall {(100*m.operating_point.recall).toFixed(2)}% · F1 {(100*m.operating_point.f1).toFixed(2)}%. NMS IoU: {m.nms_iou}.</p>}{m.limitations?.map(item=><p key={item}>{item}</p>)}<code className="pipeline-hash">{m.checkpoint_sha256}</code></section>)}</div>
 {data?.models?.filter(m=>m.target_evaluation).map(({model_type,target_evaluation:target})=><section key={model_type} className="ui-card pipeline-panel">
 <h3>Double-layer evaluation</h3>
 <p>{target.dataset} · {target.images} held-out images</p>
 <p>mAP50: {(100*target.map50).toFixed(2)}% · mAP50–95: {(100*target.map50_95).toFixed(2)}%</p>
 <p>Evaluated classes: {target.classes.map(name=>name.replaceAll('_',' ')).join(', ')}</p>
 <p>At confidence {target.operating_point.confidence}: precision {(100*target.operating_point.precision).toFixed(2)}% · recall {(100*target.operating_point.recall).toFixed(2)}%</p>
 </section>)}
 <section className="ui-card pipeline-panel"><h3>Inspection history · maintenance review</h3>
 <p>Uses the selected equipment's saved inspections. Report actual boards represented by each inspection; a single-board image represents one board.</p>
 <p>Status: {({insufficient_history:'Need six saved inspections',exposure_required:'Board counts required for all six inspections',settings_changed:'Model or confidence changed: collect six comparable inspections',review_required:'Review required: detection rate increased',no_rule_trigger:'No review rule triggered'})[data?.maintenance_trend?.status]||'Waiting for history'}</p>
 {data?.maintenance_trend?.recent_rate_per_100_boards!=null&&<p>Previous: {data.maintenance_trend.previous_rate_per_100_boards.toFixed(2)} · Recent: {data.maintenance_trend.recent_rate_per_100_boards.toFixed(2)} detections per 100 boards.</p>}
 <p>{data?.maintenance_trend?.rule}</p><p>A review alert can reflect output changes or detector errors; it does not diagnose a machine fault.</p>
 <h4>Synthetic review-alert example</h4><p>Show a fabricated increase without adding records to your inspection history.</p>
 <button className="topbar-btn topbar-btn--white" onClick={async()=>{try{setReviewDemo(await api('/maintenance/demo'));setError('');}catch(e){setError(e.message);}}}>Run synthetic review demo</button>
 {reviewDemo&&<div role="status"><p>{reviewDemo.description}</p><p>Synthetic result: {reviewDemo.trend.alert?'REVIEW REQUIRED':'No alert'} · Previous rate: {reviewDemo.trend.previous_rate_per_100_boards} · Recent rate: {reviewDemo.trend.recent_rate_per_100_boards} detections per 100 boards.</p><p>Demo observations, oldest first: {reviewDemo.inspections_oldest_first.map(row=>row.count).join(' → ')}. Nothing saved to inspection history.</p></div>}
 <h4>Simulated RUL</h4>
 {data?.rul?.rul_steps!=null&&<p>Simulated estimate: {data.rul.rul_steps>=data.rul.cap_steps?`${data.rul.cap_steps}+`:data.rul.rul_steps.toLocaleString()} inspection steps left.</p>}
 {data?.rul?.rul_steps==null&&data?.rul?.rul_cycles!=null&&<p>Previous simulated estimate: {(data.rul.rul_boards??data.rul.rul_cycles*100).toLocaleString()} boards left.</p>}
 <p>{data?.rul?.note||'Enter one board per image to estimate RUL.'}</p>
 </section>
 <section className="ui-card pipeline-panel"><h3>RUL simulation demo</h3><p>Compare the model’s estimate with the known remaining life of a simulated machine.</p>
 <button className="topbar-btn topbar-btn--white" onClick={async()=>{try{setTimeline(await api('/rul/demo'));setError('');}catch(e){setError(e.message);}}}>Run RUL demo</button>
 {error&&<p role="alert">{error}</p>}{timeline&&<><p>Simulated machine {timeline.unit} · Predictions and targets capped at {timeline.cap} steps</p><div className="pipeline-scroll"><table className="inspections-table"><thead><tr><th>Inspection step</th><th>Predicted steps left</th><th>Simulated steps left</th></tr></thead><tbody>{timeline.timeline.filter((row,index,rows)=>index%25===0||index===rows.length-1).map(row=><tr key={row.step}><td>{row.step}</td><td>{row.predicted_rul_steps}</td><td>{row.true_rul_steps}</td></tr>)}</tbody></table></div></>}
 </section></>;
}

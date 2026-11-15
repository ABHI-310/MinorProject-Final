import React, { useState } from 'react';
import { usePipeline } from '../data/PipelineContext';
export default function PipelineControls({onInspect}) {
  const {equipment,setEquipment,data,error,refresh,systemMetrics,inferenceBusy}=usePipeline();
  const [draft,setDraft]=useState(equipment);
  return <section className="ui-card pipeline-controls">
    <form onSubmit={e=>{e.preventDefault();setEquipment(draft);}}>
      <label>Equipment ID<input aria-label="Equipment ID" disabled={inferenceBusy} value={draft} onChange={e=>setDraft(e.target.value)} pattern="[A-Za-z0-9][A-Za-z0-9_.\-]{0,49}" required maxLength={50}/></label>
      <button className="topbar-btn topbar-btn--white" type="submit" disabled={inferenceBusy}>Apply equipment</button>
    </form>
    <p><strong>{systemMetrics.modelName}</strong> · Confidence cutoff {systemMetrics.confidenceThreshold}</p>
    <button className="topbar-btn topbar-btn--brown" onClick={onInspect}>Inspect a PCB</button>
    <p>Detects shorts, open circuits, spurs, mouse bites and spurious copper. Review the predicted boxes against the board image.</p>
    {!data && !error && <p role="status">Connecting to API…</p>}
    {error && <p role="alert" className="pipeline-error">API unavailable: {error}. Displayed records may be stale. <button onClick={refresh}>Retry</button></p>}
  </section>;
}

# Semester project demonstration

1. Run `.venv/bin/python scripts/manage.py app --port 8010` and open [the dashboard](http://127.0.0.1:8010/).
2. Select **Inspect a PCB** and upload the [single-layer sample](../examples/pcb_defect_002.jpg) under an equipment ID such as `DEMO_SINGLE`. Enter 1 for boards represented. At the default 0.325 cutoff, the detector matches four of five labelled defects with one extra box.
3. Upload the [double-layer sample](../examples/img62.png) under `DEMO_DOUBLE`. Enter 1 for boards represented. The same checkpoint matches all four labelled defects with one extra box.
4. Show the boxes, the saved inspection records, and the separate evaluation scores in **Settings**.
5. Click **Run synthetic review demo** to show how a rise in detections across comparable inspection batches triggers a review. Click **Run RUL demo** to compare the trained model's estimates with a simulated machine's remaining life in inspection steps.

## Evaluation

The six-class checkpoint scored 84.46% mAP50 on 148 double-layer images across short, spur, spurious copper and open circuit. It scored 83.65% mAP50 on 130 single-layer images across those classes and mouse bite. [Paired evaluation](../reports/revamp/pcb100_defectdet_missing_hole_layout11_verification.json)

For missing hole, upload [the synthetic PKU sample](../examples/pku_missing_hole_demo.jpg) from layouts by Huang and Wei ([dataset paper](https://arxiv.org/abs/1901.08204)). At the 0.40 missing-hole cutoff, the app shows three missing-hole alerts. The class produced 29 visually reviewed false alerts on the double-layer DefectDet images. [Missing-hole audit](../reports/revamp/pcb100_defectdet_missing_hole_double_layer_alert_audit.json)

The single-layer sample is `PCB_Defect/images/pcb_defect_002.jpg` from Rashid et al., *PCB-Defect*, Mendeley Data V1, CC BY 4.0. The double-layer sample is `DefectDet/images/img62.png` from Azam et al., *DefectDet*, Mendeley Data V1, CC BY 4.0. Both copies match the dataset files byte for byte. [PCB-Defect](https://doi.org/10.17632/vdj74sngvn.1) · [DefectDet](https://doi.org/10.17632/t9d9zs3bmb.1)

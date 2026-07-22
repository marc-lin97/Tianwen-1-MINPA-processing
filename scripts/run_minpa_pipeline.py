#!/usr/bin/env python
"""Command-line MINPA file -> spectra/moments/VDF workflow."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from highE.coordinates import tw1_minpa_instrument_to_orbiter_matrix, tw1_orbiter_to_mso_matrix
from highE.minpa_background import apply_background, load_background_model
from highE.minpa_io import read_records
from highE.minpa_pipeline import AttitudeSeries, interval_velocity_samples, process_records, save_product_series
from highE.minpa_plots import plot_coordinate_transform, plot_moment_timeseries, plot_spectrogram, plot_vdf_projections
from highE.minpa_processing import project_vdf
from highE.tw1_minpa import load_momag_for_day


def utc(text: str) -> float:
    value=text.replace("Z","+00:00")
    dt=datetime.fromisoformat(value)
    return (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).timestamp()


def attitude_for_time(time_s: float, root: Path) -> AttitudeSeries | None:
    day=datetime.fromtimestamp(time_s,UTC).strftime("%Y%m%d")
    source=load_momag_for_day(day,root)
    if source is None: return None
    return AttitudeSeries(source.time,source.roll,source.pitch,source.yaw,source.spacecraft_velocity_mso_km_s)


def apply_mode1_background(records, model_path: Path):
    model=load_background_model(model_path)
    if model.mode != 1 or not model.valid:
        raise ValueError("Only a valid Mode-1 background model can be applied")
    if any(record.mode != 1 or record.ion_dpf.size != 20480 for record in records):
        raise ValueError("A Mode-1 background model cannot be applied to another mode/layout")
    cube=np.stack([record.ion_dpf.reshape(40,4,16,8) for record in records])
    corrected=apply_background(cube,model).corrected_dpf
    return [replace(record,ion_dpf=corrected[i].reshape(-1)) for i,record in enumerate(records)],model


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path,help="MINPA ori .mat or released .2B file")
    parser.add_argument("--species",default="H+",choices=("H+","O+","O2+"))
    parser.add_argument("--start",required=True,help="UTC, e.g. 2021-12-31T04:00:00Z")
    parser.add_argument("--stop",required=True)
    parser.add_argument("--momag-root",type=Path,default=Path(r"D:\Data\TW-1\result\MOMAG\C\01Hz_all"))
    parser.add_argument("--output",type=Path,default=Path("outputs/minpa_example"))
    parser.add_argument("--quality-policy",choices=("none","moments","background"),default="moments")
    parser.add_argument("--energy-min",type=float,default=0.0,help="Moment/VDF lower energy bound in eV")
    parser.add_argument("--energy-max",type=float,default=float("inf"),help="Moment/VDF upper energy bound in eV")
    parser.add_argument("--reject-high-channel",action="store_true")
    parser.add_argument("--add-spacecraft-velocity",action="store_true")
    parser.add_argument("--background-model",type=Path,help="Optional valid Mode-1 model; raw input is never overwritten")
    parser.add_argument("--vdf-start",help="Optional shorter VDF interval; defaults to --start")
    parser.add_argument("--vdf-stop",help="Optional shorter VDF interval; defaults to --stop")
    parser.add_argument("--vdf-angular-samples",type=int,default=17,help="Samples per angular dimension inside each finite look cell (mature workflow default: 17)")
    parser.add_argument("--vdf-energy-samples",type=int,default=9,help="Samples between each channel's energy-table-derived logarithmic edges")
    parser.add_argument("--vdf-bins",type=int,default=120)
    args=parser.parse_args()
    start,stop=utc(args.start),utc(args.stop)
    records=read_records(args.input,start,stop)
    if not records: raise SystemExit("No records in requested interval")
    model=None
    if args.background_model:
        records,model=apply_mode1_background(records,args.background_model)
    attitude=attitude_for_time(start,args.momag_root)
    series=process_records(records,args.species,attitude=attitude,energy_min_eV=args.energy_min,energy_max_eV=args.energy_max,quality_policy=args.quality_policy,reject_high_channel=args.reject_high_channel,add_spacecraft_velocity=args.add_spacecraft_velocity)
    args.output.mkdir(parents=True,exist_ok=True)
    stem=args.output/f"mode{series.mode}_{args.species.replace('+','plus')}"
    npz,csv=save_product_series(series,stem)
    paths={"npz":str(npz),"csv":str(csv)}
    for name,plotter in (("spectrum",plot_spectrogram),("moments",plot_moment_timeseries)):
        path=args.output/f"{name}.png"; fig=plotter(series,str(path)); plt.close(fig); paths[name]=str(path)
    if series.mode != 12:
        v0=utc(args.vdf_start) if args.vdf_start else start
        v1=utc(args.vdf_stop) if args.vdf_stop else stop
        target_frame="MSO" if attitude is not None else "MINPA"
        samples=interval_velocity_samples(
            records,args.species,v0,v1,
            energy_min_eV=args.energy_min,energy_max_eV=args.energy_max,
            quality_policy=args.quality_policy,reject_high_channel=args.reject_high_channel,
            attitude=attitude,frame=target_frame,
            angular_samples=args.vdf_angular_samples,energy_samples=args.vdf_energy_samples,
        )
        rotation=None
        if attitude is not None:
            angles,_,_=attitude.nearest((v0+v1)/2)
            if angles is not None:
                roll,pitch,yaw=angles
                rotation=tw1_orbiter_to_mso_matrix(roll,pitch,yaw) @ tw1_minpa_instrument_to_orbiter_matrix()
        projection_xy=project_vdf(samples,axes=(0,1),bins=args.vdf_bins,frame=target_frame)
        vlim=float(max(abs(projection_xy.x_edges_km_s[0]),abs(projection_xy.x_edges_km_s[-1])))
        projections={
            "xy":projection_xy,
            "xz":project_vdf(samples,axes=(0,2),bins=args.vdf_bins,frame=target_frame,velocity_limit_km_s=vlim),
            "yz":project_vdf(samples,axes=(1,2),bins=args.vdf_bins,frame=target_frame,velocity_limit_km_s=vlim),
        }
        title=(f"Tianwen-1 MINPA Mode {series.mode} {args.species} finite-cell VDF; "
               f"{datetime.fromtimestamp(v0,UTC):%Y-%m-%d %H:%M:%S}--{datetime.fromtimestamp(v1,UTC):%H:%M:%S} UTC\n"
               f"{samples.record_count} records; {args.vdf_angular_samples}x{args.vdf_angular_samples} angular, "
               f"{args.vdf_energy_samples} energy samples; widths from calibrated energy-bin edges")
        path=args.output/"vdf_mso_planes.png"; fig=plot_vdf_projections(projections,str(path),title=title); plt.close(fig); paths["vdf"]=str(path)
        if rotation is not None:
            path=args.output/"coordinate_transform.png"; fig=plot_coordinate_transform(rotation,str(path)); plt.close(fig); paths["coordinates"]=str(path)
    energy_max_metadata=args.energy_max if np.isfinite(args.energy_max) else None
    metadata={"input":str(args.input),"start_utc":args.start,"stop_utc":args.stop,"species":args.species,"mode":series.mode,"energy_range_eV":[args.energy_min,energy_max_metadata],"quality_policy":args.quality_policy,"background_model":str(args.background_model) if args.background_model else None,"background_version":model.algorithm_version if model is not None else None,"coordinate_transform":"MINPA payload to spacecraft body [-V2,+V3,-V1]; spacecraft body to MSO via Rz(-Yaw) Ry(-Pitch) Rx(-Roll) from nearest MOMAG attitude","energy_width_method":"per-channel logarithmic edges inferred from adjacent calibrated energy centers; not fixed dE/E=0.15","vdf_method":{"quantity":"density-weighted reduced VDF proxy","finite_cell_sampling":{"angular_per_dimension":args.vdf_angular_samples,"energy":args.vdf_energy_samples,"energy_bounds":"per-channel logarithmic edges"},"coverage_shown_separately":True,"record_aggregation":"mean per record"},"outputs":paths}
    (args.output/"provenance.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(json.dumps(metadata,indent=2))


if __name__ == "__main__":
    main()

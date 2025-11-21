#!/usr/bin/env python3

import logging
import xarray as xr
import numpy as np
from tqdm import tqdm
import json
import torch
import argparse
from seasfire.firecastnet_lit import FireCastNetLit
from captum.attr import IntegratedGradients
import os
from captum.attr import IntegratedGradients, Saliency

logger = logging.getLogger(__name__)


def compute_sample_attributions(model, input_tensor, device, method="gradxinput"):
    """
    Compute attributions for the given input_tensor using a lighter method.

    method options:
      - 'gradxinput': fast, single backward pass
      - 'saliency': gradient magnitude
      - 'integratedgradients': slower, but more robust
    """

    input_tensor = input_tensor.clone().detach().to(device)
    input_tensor.requires_grad = True

    def forward_fn(x):
        out = model.predict_step({"x": x})
        out_flat = out.view(out.size(0), -1)
        return out_flat.sum(dim=1)

    # --- Choose attribution method ---
    if method == "integratedgradients":
        attr = IntegratedGradients(forward_fn)
        baseline = torch.zeros_like(input_tensor)
        attributions = attr.attribute(
            inputs=input_tensor, baselines=baseline, n_steps=200, internal_batch_size=1
        )
    elif method == "saliency":
        attr = Saliency(forward_fn)
        attributions = attr.attribute(inputs=input_tensor)
    else:  # gradxinput (default)
        attr = Saliency(forward_fn)
        grads = attr.attribute(inputs=input_tensor)
        attributions = grads * input_tensor  # Gradient × Input

    with torch.no_grad():
        model_out = forward_fn(input_tensor).cpu().numpy()[0]

    return attributions.detach(), float(model_out)


def main(args):
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)

    # load mean and std
    mean_std_dict_filename = f"cube_mean_std_dict_{args.target_shift}.json"
    logger.info("Opening mean-std statistics = {}".format(mean_std_dict_filename))
    with open(mean_std_dict_filename, "r") as f:
        mean_std_dict = json.load(f)

    input_vars = [
        "mslp",
        "tp",
        "vpd",
        "sst",
        "t2m_mean",
        "ssrd",
        "swvl1",
        "lst_day",
        "ndvi",
        "pop_dens",
    ]
    lsm_var = "lsm"
    static_vars = [lsm_var]
    log_preprocess_input_vars = ["tp", "pop_dens"]

    logger.info("Opening local cube zarr file: {}".format(args.cube_path))
    cube = xr.open_zarr(args.cube_path, consolidated=False)

    for var_name in log_preprocess_input_vars:
        logger.info("Log-transforming input var: {}".format(var_name))
        cube[var_name] = xr.DataArray(
            np.log(1.0 + cube[var_name].values),
            coords=cube[var_name].coords,
            dims=cube[var_name].dims,
            attrs=cube[var_name].attrs,
        )

    for static_v in static_vars:
        if "time" not in cube[static_v].dims:
            logger.info(
                "Expanding time dimension on static variable = {}.".format(static_v)
            )
            cube[static_v] = cube[static_v].expand_dims(dim={"time": cube.time}, axis=0)

    # normalize input variables
    for var in input_vars:
        var_mean = mean_std_dict[f"{var}_mean"]
        var_std = mean_std_dict[f"{var}_std"]
        cube[var] = (cube[var] - var_mean) / var_std

    # keep only needed vars
    ds = cube[input_vars + static_vars]
    ds = ds.fillna(-1)

    # shift time inputs forward in time
    logger.info(f"Shifting inputs by {args.target_shift}.")
    for var in input_vars:
        if args.target_shift > 0:
            ds[var] = ds[var].shift(time=args.target_shift, fill_value=0)

    # load model from checkpoint
    logger.info(f"Loading model from ckpt = {args.ckpt_path}")
    model = FireCastNetLit.load_from_checkpoint(
        args.ckpt_path,
        cube_path=args.cube_path,
        lsm_filter_enable=args.lsm_filter_enabled,
    )
    model.eval()

    # ensure model is on device (cuda if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.dglTo(model.device)

    logger.info(f"Will create samples for [{args.start_time}, {args.end_time}]")
    ds_selected = ds.sel(time=slice(args.start_time, args.end_time))
    ds_selected_time_indexes = ds.get_index("time").get_indexer(ds_selected["time"])

    # prepare accumulators for attribution per variable
    var_names = input_vars + static_vars
    per_sample_var_scores = (
        []
    )  # list of arrays len = number of processed samples, each array shape (C,)

    processed_count = 0
    sample_indexes_to_process = ds_selected_time_indexes

    if args.attr_samples_max is not None:
        # limit number of samples used for attribution (take first ones)
        sample_indexes_to_process = sample_indexes_to_process[: args.attr_samples_max]

    for t_index in tqdm(sample_indexes_to_process, desc="Processing samples"):
        # skip first partial windows
        if t_index < args.timeseries - 1:
            continue

        sample = ds.isel(time=slice(t_index - args.timeseries + 1, t_index + 1))

        # sample.to_array() yields shape (C, T, lat, lon) where C corresponds to var_names order
        sample_array = sample.to_array().values  # numpy

        sample_tensor = torch.tensor(sample_array, dtype=torch.float32).to(model.device)
        sample_tensor = sample_tensor.unsqueeze(0)  # shape (1, C, T, H, W)

        try:
            attributions, model_scalar = compute_sample_attributions(
                model, sample_tensor, device, args.attr_method
            )
            at_np = attributions.cpu().numpy()
            # compute mean absolute attribution per variable across time+space
            # reduce axes (batch=0), time axis=2, lat=3, lon=4 -> result shape (C,)
            mean_abs_per_var = np.mean(np.abs(at_np), axis=(0, 2, 3, 4))
            per_sample_var_scores.append(mean_abs_per_var)
            processed_count += 1
            logger.debug(
                f"Processed attribution for time index {t_index}, model_scalar={model_scalar}"
            )
        except Exception as e:
            logger.exception(
                f"Failed to compute attributions at t_index={t_index}: {e}"
            )
            # continue without attribution for this sample
            break

    if processed_count == 0:
        logger.warning("No samples processed for attribution. Exiting.")
    else:
        # stack and average across samples
        stack = np.vstack(per_sample_var_scores)  # shape (N_samples, C)
        global_importance = np.mean(stack, axis=0)  # shape (C,)

        # normalize importances to sum to 1 (optional)
        sum_imp = global_importance.sum()
        if sum_imp > 0:
            normalized_importance = global_importance / sum_imp
        else:
            normalized_importance = global_importance

        # create a dict of variable -> importance
        var_importance = {
            var: float(normalized_importance[i]) for i, var in enumerate(var_names)
        }

        logger.info("Feature importances (normalized, sum=1):")
        for var, imp in var_importance.items():
            logger.info(f"  {var:12s}: {imp:.6f}")

        # save to JSON
        out_json_path = args.save_importance_json
        if out_json_path:
            os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
            with open(out_json_path, "w") as f:
                json.dump(var_importance, f, indent=2)
            logger.info(f"Saved variable importances to {out_json_path}")

        # optionally save per-variable mean attribution maps (averaged across samples and time)
        if args.save_attr_path:
            # We will build a Dataset where each data variable is the mean attribution spatial map
            # For each processed sample we could have the full attributions; we only kept per-variable scalars.
            # Instead, we'll re-run a smaller loop to compute mean spatial attribution per variable averaged across samples,
            # but only if user requested; to avoid double work we'd need to have stored spatial attributions - not done.
            # Simpler: compute per-variable *scalar* as DataArray and save as Dataset.
            da = xr.Dataset(
                {
                    "variable_importance": xr.DataArray(
                        normalized_importance,
                        dims=("variable",),
                        coords={"variable": var_names},
                    )
                }
            )
            da.to_zarr(args.save_attr_path, mode="w")
            logger.info(
                f"Saved aggregated variable importance dataset to {args.save_attr_path}"
            )

    logger.info("Script finished.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Inference FireCastNet with feature importance"
    )
    parser.add_argument(
        "--cube-path",
        metavar="KEY",
        type=str,
        action="store",
        dest="cube_path",
        default="cube.zarr",
        help="Cube path",
    )
    parser.add_argument(
        "--ckpt-path",
        metavar="KEY",
        type=str,
        action="store",
        dest="ckpt_path",
        default="best.ckpt",
        help="Checkpoint path",
    )
    parser.add_argument(
        "--target-shift",
        metavar="KEY",
        type=int,
        action="store",
        dest="target_shift",
        default=1,
        help="Target shift",
    )
    parser.add_argument(
        "--timeseries",
        metavar="KEY",
        type=int,
        action="store",
        dest="timeseries",
        default=24,
        help="Timeseries length",
    )
    parser.add_argument(
        "--start-time",
        metavar="KEY",
        type=str,
        action="store",
        dest="start_time",
        default="2019-01-01",
        help="Start time",
    )
    parser.add_argument(
        "--end-time",
        metavar="KEY",
        type=str,
        action="store",
        dest="end_time",
        default="2020-01-01",
        help="End time",
    )
    parser.add_argument(
        "--save-importance-json",
        metavar="PATH",
        type=str,
        action="store",
        dest="save_importance_json",
        default="variable_importances.json",
        help="Path to JSON file where normalized variable importances will be saved.",
    )
    parser.add_argument(
        "--save-attr-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="save_attr_path",
        default=None,
        help="Optional: path to save aggregated attributions (zarr).",
    )
    parser.add_argument(
        "--attr-samples-max",
        metavar="N",
        type=int,
        action="store",
        dest="attr_samples_max",
        default=None,
        help="Optional: maximum number of samples to use for computing attribution (useful to speed up).",
    )
    parser.add_argument(
        "--attr-method",
        metavar="METHOD",
        type=str,
        choices=["gradxinput", "saliency", "integratedgradients"],
        default="gradxinput",
        help="Attribution method to use (default: gradxinput).",
    )
    parser.add_argument("--lsm-filter", dest="lsm_filter_enabled", action="store_true")
    parser.add_argument(
        "--no-lsm-filter", dest="lsm_filter_enabled", action="store_false"
    )
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--no-debug", dest="debug", action="store_false")
    parser.set_defaults(lsm_filter_enabled=False, debug=False)
    args = parser.parse_args()

    main(args)

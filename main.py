from __future__ import annotations

from pathlib import Path

import config_model
import config_param
import gen_predictions


def _progress_update(progress: float) -> None:
    pct = int(progress * 100)
    bucket = min(100, (pct // 10) * 10)
    if not hasattr(_progress_update, "last_bucket"):
        _progress_update.last_bucket = -10
    if bucket > _progress_update.last_bucket:
        _progress_update.last_bucket = bucket
        print(f"Predictor generation: {bucket}% complete")


def _validate_dimensions() -> None:
    n_locations_hosp = int(config_param.hosp_dat.shape[0])
    n_locations_pop = int(len(config_param.popu))
    n_locations_names = int(len(config_param.state_abbr))

    if n_locations_hosp != n_locations_pop or n_locations_hosp != n_locations_names:
        raise ValueError(
            "Location mismatch: target data rows and location metadata must have the same number of locations."
        )


def main() -> None:
    output_path = Path(config_param.cli_output_path)
    output_format = config_param.cli_output_format
    ensemble_name = config_param.ensemble_method

    _validate_dimensions()

    print("Generating predictors...")
    all_preds, hosp_dat = gen_predictions.generate_all_preds(progress_callback=_progress_update)

    print(f"Generating ensemble predictions using: {ensemble_name}")
    ensemble = config_model.create_ensemble(ensemble_name)
    preds = gen_predictions.generate_preds(all_preds, hosp_dat, ensemble=ensemble)

    output_df = gen_predictions.predictions_to_long_df(preds)
    gen_predictions.save_predictions_df(output_df, output_path, output_format)

    print(f"Saved {len(output_df)} prediction rows to {output_path.resolve()}")


if __name__ == "__main__":
    main()

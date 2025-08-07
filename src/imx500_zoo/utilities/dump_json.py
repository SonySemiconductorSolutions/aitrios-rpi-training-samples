import json
import os
import sys
import datetime


def dump_json(file_name, solution, validate_results=None, is_path_raw=False):
    dump_data = _get_dump_data(solution, validate_results)

    fpath = file_name if is_path_raw else os.path.join("results", f"{file_name}.json")
    dpath = os.path.dirname(fpath)
    if len(dpath) > 0:
        os.makedirs(dpath, exist_ok=True)
    with open(fpath, mode="wt", encoding="utf-8") as f:
        json.dump(dump_data, f, indent=4)


def _get_dump_data(solution, validate_results):
    config = solution.config
    results = config.results

    path = config["PATH"]
    dataset_path = path.get("DATASET", None)
    if solution.config["SOLUTION"]["FRAMEWORK"] == "keras":
        annotation_path = path.get("ANNOTATION_FILE", None)
        float_model_path = solution.config["PATH"]["KERAS"]
        quantized_model_path = solution.config["PATH"]["QUANTIZED_KERAS"]

    elif solution.config["SOLUTION"]["FRAMEWORK"] == "pytorch":
        annotation_path = None
        float_model_path = solution.config["PATH"]["ONNX"]
        quantized_model_path = solution.config["PATH"]["QUANTIZED_ONNX"]

    else:
        print("This framework has not supported yet")
        exit()

    path_dict = {
        "DATASET": dataset_path,
        "ANNOTATION_FILE": annotation_path,
        "MODEL": float_model_path,
        "QUANTIZED_MODEL": quantized_model_path,
    }

    val_results_dict = {}
    if validate_results is None:
        if len(results.valid) > 0:
            val_results_dict = results.valid
    else:
        val_results_before_quant = validate_results[0]
        val_results_after_quant = validate_results[1]

        val_results_dict = {
            "MODEL": val_results_before_quant,
            "QUANTIZED_MODEL": val_results_after_quant,
        }

    dump_data = {
        "RUN": _dump_run(solution),
        "PATH": path_dict,
        "VAL_RESULTS": val_results_dict,
    }

    if len(results.train) > 0:
        dump_data["TRAIN_RESULTS"] = results.train

    if len(results.quant) > 0:
        dump_data["QUANT_RESULTS"] = results.quant

    return dump_data


def _dump_run(solution):
    run = datetime.datetime.today() - solution.start_date
    fp = solution.config_file_path
    ini = open(fp, "r").read() if os.path.isfile(fp) else ""

    ret = {
        "ARGV": sys.argv,
        "PLATFORM": sys.platform,
        "CWD": os.getcwd(),
        "START_DATE": _date2str(solution.start_date),
        "RUN_MINITE": int(run.total_seconds() / 60),
        "INI": ini,
    }

    if "TRAINER" in solution.config:
        conf = solution.config["TRAINER"].get("CONFIG", None)
        if conf is not None:
            ret[conf] = open(conf, "r").read() if os.path.isfile(conf) else ""

    return ret


def _date2str(date):
    return date.strftime("%Y/%m/%d %H:%M:%S")

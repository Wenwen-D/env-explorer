from data.csv_explore.csv_sampler_base import LogLinearModel, FilenameToFormatModel
import csv
from pathlib import Path

# def extract_features_dynamic(filename: str):
#     lower = filename.lower()
#     features = {"bias": 1.0}
#     for feat_name, tokens in FEATURE_ON_DICT.items():
#         features[feat_name] = any(token in lower for token in tokens)
#     return features
def build_model_min():
    delimiters = [",", ";", "\t"]
    quotechars = ['"', "'"]

    features = ["has_eu", "has_tsv", "has_sas"]  # + bias

    Theta_delim = {
        ",":  {"bias": 2.0, "has_eu": -1.5, "has_tsv": -3.0},
        ";":  {"bias": 0.0, "has_eu": +1.3},
        "\t": {"bias":0.0, "has_tsv": +5.0, "has_eu": -1.0},
    }
    Theta_quote = {
        '"': {"bias": 0.7, "has_sas": -2.0},
        "'": {"bias": -0.1, "has_sas": +2.0},
    }

    def extract_features(filename: str):
        lower = filename.lower()
        return {
            "bias": 1.0,
            "has_eu": "eu" in lower,
            "has_tsv": "tsv" in lower or "tab" in lower,
            "has_sas": "sas" in lower,
        }
    model_dict = {
        "Delimiter": LogLinearModel("Delimiter", delimiters, Theta_delim),
        "Quotechar": LogLinearModel("Quotechar", quotechars, Theta_quote),
    }
    feature_dict =  {
        "has_eu": ['eu'],
        "has_tsv": ['tsv'],
        "has_sas": ['sas'],
    }
    return FilenameToFormatModel(extract_features, model_dict), feature_dict


# def build_model_more_easy(verify = False):
#     delimiters = [",", ";", "\t"]
#     quotechars = ['"', "'"]
#     encodings = ["utf-8", "utf-16"]

#     features = ["has_eu", "has_tsv", "has_sas", "has_cn"]  # + bias

#     Theta_delim = {
#         ",":  {"bias": 2.0, "has_eu": -1.5, "has_tsv": -3.0},
#         ";":  {"bias": 0.0, "has_eu": +1.3},
#         "\t": {"bias":0.0, "has_tsv": +5.0, "has_eu": -1.0},
#     }
#     Theta_quote = {
#         '"': {"bias": 0.7, "has_sas": -2.0},
#         "'": {"bias": -0.1, "has_sas": +2.0},
#     }
#     Theta_encoding = {
#         "utf-8":  {"bias": 1.0, "has_eu": -0.7},
#         "utf-16": {"bias":-1.0, "has_cn": +3.0},  # strong boost for has_cn
#     }


#     def extract_features_more(filename: str):
#         """Simple binary features from filename tokens."""
#         lower = filename.lower()
#         return {
#             "bias": 1.0,
#             "has_eu": "eu" in lower,
#             "has_tsv": "tsv" in lower,
#             "has_sas": "sas" in lower,
#             "has_cn": any(tag in lower for tag in ["cn", "zh"]),
#         }
#     model_dict = {
#         "Delimiter": LogLinearModel("Delimiter", delimiters, Theta_delim),
#         "Quotechar": LogLinearModel("Quotechar", quotechars, Theta_quote),
#         "Encoding": LogLinearModel("Encoding", encodings, Theta_encoding),
#     }
#     feature_dict =  {
#         "has_eu": ['eu'],
#         "has_tsv": ['tsv'],
#         "has_sas": ['sas'],
#         "has_cn": ['cn', 'zh'],
#     }
#     if verify:
#         out_path = Path("feature_influence.csv")
#         with open(out_path, "w", newline="") as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(["Feature", "Model", "Option", "OFF_Prob", "ON_Prob", "Diff"])

#             print(f"=== Writing feature influence summary to {out_path} ===")

#             for feat in features:
#                 print(f"\n>>> Feature: {feat}")
#                 feats_off = {f: 0.0 for f in features}
#                 feats_off["bias"] = 1.0
#                 feats_on = feats_off.copy()
#                 feats_on[feat] = 1.0

#                 for name, model in model_dict.items():
#                     p_off = model.predict_proba(feats_off)
#                     p_on = model.predict_proba(feats_on)

#                     print(f"  {name}:")
#                     for opt in model.options:
#                         diff = p_on[opt] - p_off[opt]
#                         print(f"    {repr(opt):8s} | OFF: {p_off[opt]:5.2f} | ON: {p_on[opt]:5.2f} | Δ: {diff:+.2f}")

#                         # Write row to CSV
#                         writer.writerow([feat, name, opt, round(p_off[opt], 4), round(p_on[opt], 4), round(diff, 4)])

#                 print("-" * 60)
#     return FilenameToFormatModel(extract_features_more, model_dict), feature_dict

def build_model_more(verify = False):
    delimiters = [",", ";", "\t"]
    quotechars = ['"', "'"]
    encodings = ["utf-8", "utf-16"]

    features = ["has_eu", "has_tsv", "has_sas", "has_cn"]  # + bias

    Theta_delim = {
        ",":  {"bias": 1.1, "has_eu": -0.4, "has_tsv": -0.5},
        ";":  {"bias": 0.9, "has_eu": +0.45, "has_tsv": -0.2},
        "\t": {"bias":0.8, "has_tsv": +0.9,},
    }
    Theta_quote = {
        '"': {"bias": 1, "has_sas": -1.0, "has_cn": +1.0},
        "'": {"bias": 0.6, "has_sas": +1.0},
    }
    Theta_encoding = {
        "utf-8":  {"bias": 1.0, "has_eu": -0.7},
        "utf-16": {"bias":0.9, "has_cn": +1.0},  # strong boost for has_cn
    }


    def extract_features_more(filename: str):
        """Simple binary features from filename tokens."""
        lower = filename.lower()
        return {
            "bias": 1.0,
            "has_eu": "eu" in lower,
            "has_tsv": "tsv" in lower,
            "has_sas": "sas" in lower,
            "has_cn": any(tag in lower for tag in ["cn", "zh"]),
        }
    model_dict = {
        "Delimiter": LogLinearModel("Delimiter", delimiters, Theta_delim),
        "Quotechar": LogLinearModel("Quotechar", quotechars, Theta_quote),
        "Encoding": LogLinearModel("Encoding", encodings, Theta_encoding),
    }
    feature_dict =  {
        "has_eu": ['eu'],
        "has_tsv": ['tsv'],
        "has_sas": ['sas'],
        "has_cn": ['cn', 'zh'],
    }
    if verify:
        out_path = Path("feature_influence.csv")
        with open(out_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Feature", "Model", "Option", "OFF_Prob", "ON_Prob", "Diff"])

            print(f"=== Writing feature influence summary to {out_path} ===")

            for feat in features:
                print(f"\n>>> Feature: {feat}")
                feats_off = {f: 0.0 for f in features}
                feats_off["bias"] = 1.0
                feats_on = feats_off.copy()
                feats_on[feat] = 1.0

                for name, model in model_dict.items():
                    p_off = model.predict_proba(feats_off)
                    p_on = model.predict_proba(feats_on)

                    print(f"  {name}:")
                    for opt in model.options:
                        diff = p_on[opt] - p_off[opt]
                        print(f"    {repr(opt):8s} | OFF: {p_off[opt]:5.2f} | ON: {p_on[opt]:5.2f} | Δ: {diff:+.2f}")

                        # Write row to CSV
                        writer.writerow([feat, name, opt, round(p_off[opt], 4), round(p_on[opt], 4), round(diff, 4)])

                print("-" * 60)
    return FilenameToFormatModel(extract_features_more, model_dict), feature_dict


def build_model_independent(verify = False):
    delimiters = [",", ";", "\t"]
    quotechars = ['"', "'"]
    skiprows=[0, 1]

    features = ["has_eu", "has_tsv", "has_sas", "has_cn"]  # + bias

    Theta_delim = {
        ",":  {"bias": 2.0, "has_eu": -1.5, "has_tsv": -3.0},
        ";":  {"bias": 0.0, "has_eu": +1.3},
        "\t": {"bias":0.0, "has_tsv": +5.0, "has_eu": -1.0},
    }
    Theta_quote = {
        '"': {"bias": 0.7, "has_sas": -2.0},
        "'": {"bias": -0.1, "has_sas": +2.0},
    }
    Theta_skiprows = {
        0:  {"bias": 1.0, "has_eu": -0.7},
        1:  {"bias":-1.0, "has_cn": +3.0}, 
    }
    


    def extract_features_more(filename: str):
        """Simple binary features from filename tokens."""
        lower = filename.lower()
        return {
            "bias": 1.0,
            "has_eu": "eu" in lower,
            "has_tsv": "tsv" in lower,
            "has_sas": "sas" in lower,
            "has_cn": any(tag in lower for tag in ["cn", "zh"]),
        }
    model_dict = {
        "Delimiter": LogLinearModel("Delimiter", delimiters, Theta_delim),
        "Quotechar": LogLinearModel("Quotechar", quotechars, Theta_quote),
        "Skiprows": LogLinearModel("Skiprows", skiprows, Theta_skiprows),
    }
    feature_dict =  {
        "has_eu": ['eu'],
        "has_tsv": ['tsv'],
        "has_sas": ['sas'],
        "has_cn": ['cn', 'zh'],
    }
    if verify:
        out_path = Path("feature_influence.csv")
        with open(out_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Feature", "Model", "Option", "OFF_Prob", "ON_Prob", "Diff"])

            print(f"=== Writing feature influence summary to {out_path} ===")

            for feat in features:
                print(f"\n>>> Feature: {feat}")
                feats_off = {f: 0.0 for f in features}
                feats_off["bias"] = 1.0
                feats_on = feats_off.copy()
                feats_on[feat] = 1.0

                for name, model in model_dict.items():
                    p_off = model.predict_proba(feats_off)
                    p_on = model.predict_proba(feats_on)

                    print(f"  {name}:")
                    for opt in model.options:
                        diff = p_on[opt] - p_off[opt]
                        print(f"    {repr(opt):8s} | OFF: {p_off[opt]:5.2f} | ON: {p_on[opt]:5.2f} | Δ: {diff:+.2f}")

                        # Write row to CSV
                        writer.writerow([feat, name, opt, round(p_off[opt], 4), round(p_on[opt], 4), round(diff, 4)])

                print("-" * 60)
    return FilenameToFormatModel(extract_features_more, model_dict), feature_dict

def build_model_independent_hard(verify = False):
    delimiters = [",", ";", "\t"]
    quotechars = ['"', "'"]
    skiprows=[0, 1]

    features = ["has_eu", "has_tsv", "has_sas", "has_cn"]  # + bias

    Theta_delim = {
        ",":  {"bias": 1.7, "has_eu": -1.5,},
        ";":  {"bias": 0, "has_eu": +2},
        "\t": {"bias":0.5, "has_dat": +2.0, "has_eu": -1.0, "has_tsv": +2.0},
    }
    Theta_quote = {
        '"': {"bias": 0.7, "has_sas": -1.0},
        "'": {"bias": 0.2, "has_sas": +1.0},
    }
    Theta_skiprows = {
        0:  {"bias": 1.5, "has_eu": -0.7},
        1:  {"bias":0, "has_cn": +1.0}, 
    }
    


    def extract_features_more(filename: str):
        """Simple binary features from filename tokens."""
        lower = filename.lower()
        return {
            "bias": 1.0,
            "has_eu": "eu" in lower,
            "has_tsv": "tsv" in lower,
            "has_sas": "sas" in lower,
            "has_cn": any(tag in lower for tag in ["cn", "zh"]),
        }
    model_dict = {
        "Delimiter": LogLinearModel("Delimiter", delimiters, Theta_delim),
        "Quotechar": LogLinearModel("Quotechar", quotechars, Theta_quote),
        "Skiprows": LogLinearModel("Skiprows", skiprows, Theta_skiprows),
    }
    feature_dict =  {
        "has_eu": ['eu'],
        "has_tsv": ['tsv'],
        "has_sas": ['sas'],
        "has_cn": ['cn', 'zh'],
    }
    if verify:
        out_path = Path("feature_influence.csv")
        with open(out_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Feature", "Model", "Option", "OFF_Prob", "ON_Prob", "Diff"])

            print(f"=== Writing feature influence summary to {out_path} ===")

            for feat in features:
                print(f"\n>>> Feature: {feat}")
                feats_off = {f: 0.0 for f in features}
                feats_off["bias"] = 1.0
                feats_on = feats_off.copy()
                feats_on[feat] = 1.0

                for name, model in model_dict.items():
                    p_off = model.predict_proba(feats_off)
                    p_on = model.predict_proba(feats_on)

                    print(f"  {name}:")
                    for opt in model.options:
                        diff = p_on[opt] - p_off[opt]
                        print(f"    {repr(opt):8s} | OFF: {p_off[opt]:5.2f} | ON: {p_on[opt]:5.2f} | Δ: {diff:+.2f}")

                        # Write row to CSV
                        writer.writerow([feat, name, opt, round(p_off[opt], 4), round(p_on[opt], 4), round(diff, 4)])

                print("-" * 60)
    return FilenameToFormatModel(extract_features_more, model_dict), feature_dict

# def build_model_independent_hard(verify = False):
#     delimiters = [",", ";", "\t"]
#     quotechars = ['"', "'"]
#     skiprows=[0, 1]

#     features = ["has_eu", "has_dat", "has_sas", "has_cn"]  # + bias

#     Theta_delim = {
#         ",":  {"bias": 1.0, "has_eu": -1.5,},
#         ";":  {"bias": 0, "has_eu": +2},
#         "\t": {"bias":0.5, "has_dat": +2.0, "has_eu": -1.0},
#     }
#     Theta_quote = {
#         '"': {"bias": 0.7, "has_sas": -1.0},
#         "'": {"bias": 0.2, "has_sas": +1.0},
#     }
#     Theta_skiprows = {
#         0:  {"bias": 0.2, "has_eu": -0.7},
#         1:  {"bias":0, "has_cn": +1.0}, 
#     }
    


#     def extract_features_more(filename: str):
#         """Simple binary features from filename tokens."""
#         lower = filename.lower()
#         return {
#             "bias": 1.0,
#             "has_eu": "eu" in lower,
#             "has_dat": any(tag in lower for tag in ["dat", "log"]),
#             "has_sas": "sas" in lower,
#             "has_cn": any(tag in lower for tag in ["cn", "zh"]),
#         }
#     model_dict = {
#         "Delimiter": LogLinearModel("Delimiter", delimiters, Theta_delim),
#         "Quotechar": LogLinearModel("Quotechar", quotechars, Theta_quote),
#         "Skiprows": LogLinearModel("Skiprows", skiprows, Theta_skiprows),
#     }
#     feature_dict =  {
#         "has_eu": ['eu'],
#         "has_dat": ['dat', 'log'],
#         "has_sas": ['sas'],
#         "has_cn": ['cn', 'zh'],
#     }
#     if verify:
#         out_path = Path("feature_influence.csv")
#         with open(out_path, "w", newline="") as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(["Feature", "Model", "Option", "OFF_Prob", "ON_Prob", "Diff"])

#             print(f"=== Writing feature influence summary to {out_path} ===")

#             for feat in features:
#                 print(f"\n>>> Feature: {feat}")
#                 feats_off = {f: 0.0 for f in features}
#                 feats_off["bias"] = 1.0
#                 feats_on = feats_off.copy()
#                 feats_on[feat] = 1.0

#                 for name, model in model_dict.items():
#                     p_off = model.predict_proba(feats_off)
#                     p_on = model.predict_proba(feats_on)

#                     print(f"  {name}:")
#                     for opt in model.options:
#                         diff = p_on[opt] - p_off[opt]
#                         print(f"    {repr(opt):8s} | OFF: {p_off[opt]:5.2f} | ON: {p_on[opt]:5.2f} | Δ: {diff:+.2f}")

#                         # Write row to CSV
#                         writer.writerow([feat, name, opt, round(p_off[opt], 4), round(p_on[opt], 4), round(diff, 4)])

#                 print("-" * 60)
#     return FilenameToFormatModel(extract_features_more, model_dict), feature_dict

build_model_independent_hard(True)
min_model_instance, feature_dict = build_model_independent_hard()
filenames = [
        "survey_results_cn_2024.tsv",
        "financial_report_eu_2023.csv",
        "marketing_data_us.csv",
        "clinical_sas_export.csv",
        "student_scores_tsv_2022.tsv",
        "china_utf16_sales.csv",
        "population_france_2025.csv",
    ]
for filename in filenames[:1]:
    print(f"\n--- Analyzing filename: {filename} ---")
    features = min_model_instance.feature_fn(filename)

    p_delim    = min_model_instance.models["Delimiter"].predict_proba(features)
    p_quote    = min_model_instance.models["Quotechar"].predict_proba(features)
    p_skiprows = min_model_instance.models["Skiprows"].predict_proba(features)

    print("Features:", features)
    print("\nDelimiter probs:", p_delim)
    print("Quotechar probs:", p_quote)
    print("Skiprows probs:", p_skiprows)
    print("-" * 60)
    feats, probs = min_model_instance.predict_all(filename)
    sampled = min_model_instance.sample_all(filename)
    print(f"\n=== {filename} ===")
    print("Features:", feats)
    for name, dist in probs.items():
        print(f"{name}: {dist}")
    print("→ Sampled choice:", sampled)



//! File › Export as Python: the workspace as a folder of Python that does what
//! the canvas does – one `run.py` to execute, the plan the canvas makes, the
//! datasets, the trainer, the model's native tools and a README.

use std::path::{Path, PathBuf};

use magi_core::magic::{InputSpec, Layout, ModelKind};
use magi_core::nodes::studio::{self, PlanFiles};

use super::{MagicDoc, file_stem};

const RUN_PY: &str = r#""""{name} – trains the model of a MacroMagi Magic workspace, as MacroMagi does.

    python run.py              train it, then write the Magi model ({stem}.szi)
    python run.py --package    only write the model file (after training)

Needs Python 3.10+ with PyTorch (pip install -r requirements.txt; for an NVIDIA
graphics card, PyTorch's CUDA build from pytorch.org). It prints its progress as
one JSON object per line, as MacroMagi reads it.
"""
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, os.path.join(HERE, "runtime"))


def train():
    from magi_train import train as trainer
    return trainer.main("plan.json")


def family(parameters):
    return "Magi-F" if parameters <= 1_000_000_000 else "Magi-S" if parameters <= 30_000_000_000 else "Magi-T"


def package():
    """The Magi model as one .szi file – a zip archive, as MacroMagi writes it: model.json (the
    description), the weights (stored as they are), the ONNX model if one was made, and the
    native tools."""
    model = json.load(open("model.json", encoding="utf-8"))
    report = json.load(open(os.path.join("out", "report.json"), encoding="utf-8"))
    model["trained"] = {k: report.get(k) for k in ("stages", "steps", "loss", "parameters", "compacts")}
    # Tokens the trainer learned where it ran, from a corpus.
    learned = os.path.join("out", "tokenizer.json")
    if os.path.isfile(learned) and model.get("text"):
        model["text"]["tokenizer"] = json.load(open(learned, encoding="utf-8"))
    files = {}
    for name in ("weights.safetensors", "model.onnx"):
        if os.path.isfile(os.path.join("out", name)):
            files["weights" if name.endswith("safetensors") else "onnx"] = name
    if os.path.isdir("nt"):
        files["native_tools"] = "nt"
    manifest = {"format": "magi.szi", "version": 2, "app": "MacroMagi {version} (Python export)",
                "family": family(report.get("parameters") or 0), "files": files}
    manifest.update(model)
    path = os.path.join(HERE, "{stem}.szi")
    with zipfile.ZipFile(path + ".part", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("model.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        for name in ("weights.safetensors", "model.onnx"):
            if name in files.values():
                archive.write(os.path.join("out", name), name, compress_type=zipfile.ZIP_STORED)
        for root, _, names in os.walk("nt"):
            for name in names:
                source = os.path.join(root, name)
                archive.write(source, os.path.relpath(source, HERE).replace(os.sep, "/"))
        archive.writestr("README.md", '# ' + model.get("name", "{stem}") + "\n\nA Magi model in the .szi format "
                         "(a zip archive), trained by a workspace MacroMagi exported as Python. Open it in MacroMagi "
                         "(File › Open, or the AI Agent window).\n")
    os.replace(path + ".part", path)
    print(json.dumps({{"event": "log", "message": "Wrote " + path}}))


if __name__ == "__main__":
    if "--package" not in sys.argv:
        code = train()
        if code:
            sys.exit(code)
    package()
"#;

/// Writes the workspace as Python into a new folder in `into`; returns the folder. `compile` asks the
/// trainer to try compiled layers (experimental: Linux only, kept only when faster with the same loss).
pub fn export(doc: &MagicDoc, runtime: &Path, into: &Path, compile: bool) -> Result<PathBuf, String> {
    let pipeline = studio::pipeline(&doc.studio.graph)?;
    if !pipeline.trains() {
        return Err("The canvas does not train a model: add training stages first.".into());
    }
    let stem = file_stem(&doc.title());
    let dir = into.join(format!("{stem} (Python)"));
    let data = dir.join("data");
    std::fs::create_dir_all(&data).map_err(|e| format!("The folder could not be made: {e}"))?;
    let write = |path: &Path, bytes: &[u8]| std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()));

    // The datasets, by the paths the plan gives the trainer.
    let mut written = std::collections::BTreeMap::new();
    for (index, (name, bytes)) in doc.file.data.iter().enumerate() {
        if let Some(file) = name.strip_prefix("data/text/") {
            let relative = format!("data/{index}-{}", file.replace('/', "-"));
            write(&dir.join(&relative), bytes)?;
            written.insert(name.clone(), relative);
        }
    }
    // The weights to start from.
    let mut weights = None;
    if pipeline.entry.kind == studio::kinds::BASE {
        let params: studio::BaseParams = pipeline.entry.params_or_default();
        let bytes = if params.path.trim().is_empty() {
            doc.file.weights.clone()
        } else {
            magi_core::szi::open(Path::new(params.path.trim())).map_err(|e| e.to_string())?.weights
        };
        write(&data.join("base.safetensors"), &bytes)?;
        weights = Some("data/base.safetensors".to_owned());
    }
    // The pictures, prepared as the trainer reads them (vision).
    let kind = pipeline.kind(&doc.file.model);
    let mut pictures = None;
    if kind == ModelKind::Vision {
        let side = pipeline.architecture.context.clamp(8, 1024);
        let spec = InputSpec {
            width: side,
            height: side,
            layout: Layout::Nchw,
            softmax: true,
            ..doc.file.model.input.clone()
        };
        super::train::write_pictures(&doc.file, &spec, &data.join("pictures.bin"))?;
        pictures = Some("data/pictures.bin".to_owned());
    }
    let files = PlanFiles {
        dataset: &|name: &str| written.get(name).cloned(),
        weights,
        pictures,
        out_dir: "out".into(),
        devices: Vec::new(),
    };
    let mut plan = studio::plan(&pipeline, &doc.studio.graph, &doc.file.model, &files)?;
    // Whatever graphics cards the computer it runs on has.
    plan["devices"] = serde_json::json!("auto");
    if compile {
        plan["compile"] = serde_json::json!(true);
    }
    write(&dir.join("plan.json"), &serde_json::to_vec_pretty(&plan).unwrap_or_default())?;
    // onnx only for a model that is written as ONNX: a vision model, or one the native engine cannot run.
    let needs_onnx = kind == ModelKind::Vision || plan["onnx"].as_bool().unwrap_or(true);
    // A corpus read where it trains needs Hugging Face's tools to fetch it and cut it into tokens.
    let reads_corpus = plan["datasets"]
        .as_object()
        .is_some_and(|d| d.values().filter_map(|v| v.as_array()).flatten().any(|f| f.get("corpus").is_some()));

    // The model's description, as the package will carry it.
    let mut model = doc.file.model.clone();
    model.kind = kind;
    model.architecture = Some(pipeline.architecture.clone());
    model.studio = None;
    model.datasets.clear();
    model.training.clear();
    model.finetune.clear();
    model.package = None;
    write(&dir.join("model.json"), &serde_json::to_vec_pretty(&model).unwrap_or_default())?;
    if let Some(text) = &model.text
        && text.tools.iter().any(|t| t.native)
    {
        magi_core::native_tools::write_folder(&text.tools, &doc.file.data, &dir.join("nt"))
            .map_err(|e| e.to_string())?;
    }

    // The trainer.
    copy_dir(&runtime.join("magi_train"), &dir.join("runtime").join("magi_train"))?;
    let run = RUN_PY
        .replace("{name}", &doc.title())
        .replace("{stem}", &stem)
        .replace("{version}", magi_core::VERSION)
        .replace("{{", "{")
        .replace("}}", "}");
    write(&dir.join("run.py"), run.as_bytes())?;
    let mut requirements = "torch\n".to_owned();
    if needs_onnx {
        requirements.push_str("onnx\n");
    }
    if reads_corpus {
        requirements.push_str("huggingface_hub\npyarrow\ntokenizers\nnumpy\n");
    }
    write(&dir.join("requirements.txt"), requirements.as_bytes())?;
    write(&dir.join("colab.ipynb"), colab_notebook(&format!("{stem} (Python)"), &stem, &requirements).as_bytes())?;
    let stages: Vec<String> =
        pipeline.stages.iter().map(|s| s.kind.trim_start_matches("magic.").replace('_', " ")).collect();
    let readme = format!(
        "# {name} – the workspace as Python\n\n\
         This folder does what the Magic tab's canvas does, without MacroMagi: it trains **{name}** and writes \
         its Magi package.\n\n\
         ```bash\npip install -r requirements.txt\npython run.py\n```\n\n\
         For an NVIDIA graphics card, install PyTorch's CUDA build first (see pytorch.org). Progress is printed \
         as one JSON object per line.\n\n\
         ## What is here\n\n\
         | | |\n| --- | --- |\n\
         | `run.py` | Runs the training, then writes `{stem}.szi` – the model in Magi's format, one file with its \
         weights and its native tools. Open it in MacroMagi to use it. |\n\
         | `plan.json` | The canvas as the trainer reads it: the architecture ({layers} layers, {width} wide, {context} \
         tokens of context), the stages ({stages}), their datasets and optimizers. |\n\
         | `model.json` | The model's description: how it reads and writes, its tools, thinking and effort. |\n\
         | `data/` | The datasets{pictures}. |\n\
         | `nt/` | The model's native tools, when it has some. |\n\
         | `runtime/magi_train/` | MacroMagi's trainer, unchanged. |\n\n\
         Change `plan.json` to change the training: the steps, the batch, the optimizer, the layers.\n\n\
         ## Training in the cloud\n\n\
         Any computer with Python and an NVIDIA graphics card can run it – a rented GPU server too: copy this \
         folder there, run the two lines above, and copy `{stem}.szi` back.\n\n\
         On **Google Colab**: zip this folder, open `colab.ipynb` in Colab (File › Upload notebook), choose a GPU \
         (Runtime › Change runtime type), and run its cells. It asks for the zip once and keeps everything on \
         Google Drive, so when a session ends you run the cells again and it goes on; at the end it downloads \
         `{stem}.szi` (a copy stays on Drive). Open it in MacroMagi.\n",
        name = doc.title(),
        layers = pipeline.architecture.layers.len(),
        width = pipeline.architecture.width,
        context = pipeline.architecture.context,
        stages = stages.join(", "),
        pictures = if kind == ModelKind::Vision { " and the pictures, prepared" } else { "" },
    );
    write(&dir.join("README.md"), readme.as_bytes())?;
    Ok(dir)
}

/// A Colab notebook: the zipped folder (uploaded once, then kept on Google Drive with the run and the
/// corpora), install, train, download the Magi model.
fn colab_notebook(folder: &str, stem: &str, requirements: &str) -> String {
    let cell = |kind: &str, lines: &[&str]| {
        let source: Vec<String> = lines
            .iter()
            .enumerate()
            .map(|(n, l)| if n + 1 < lines.len() { format!("{l}\n") } else { (*l).to_owned() })
            .collect();
        let mut cell = serde_json::json!({ "cell_type": kind, "metadata": {}, "source": source });
        if kind == "code" {
            cell["execution_count"] = serde_json::Value::Null;
            cell["outputs"] = serde_json::json!([]);
        }
        cell
    };
    let quoted = |text: &str| text.replace('\\', "\\\\").replace('\'', "\\'");
    let (folder_q, stem_q) = (quoted(folder), quoted(stem));
    // Everything on Google Drive – the project, the run and the corpora – so a session that ends
    // loses nothing: run the cells again and it goes on, with no upload.
    let project = format!(
        "from google.colab import drive, files\nimport os, shutil, zipfile\ndrive.mount('/content/drive')\n\
         keep = '/content/drive/MyDrive/MacroMagi/{stem_q}'\ncode = keep + '/project'\n\
         NEW_ZIP = False  # True: upload a new export (a saved run goes on only when its plan is the same)\n\
         os.chdir('/content')\nshutil.rmtree('{folder_q}', ignore_errors=True)\n\
         if NEW_ZIP or not os.path.exists(code + '/run.py'):\n    uploaded = files.upload()\n\
         \x20   zipfile.ZipFile(next(iter(uploaded))).extractall('.')\n    shutil.rmtree(code, ignore_errors=True)\n\
         \x20   shutil.copytree('{folder_q}', code, ignore=shutil.ignore_patterns('out', '__pycache__'))\n\
         else:\n    shutil.copytree(code, '{folder_q}')\n\
         os.chdir('{folder_q}')\nos.makedirs(keep + '/out', exist_ok=True)\n\
         if os.path.isdir('out') and not os.path.islink('out'):\n    shutil.rmtree('out')\n\
         if not os.path.exists('out'):\n    os.symlink(keep + '/out', 'out')\n\
         os.environ['MAGI_CORPUS_DIR'] = keep + '/corpus'\nprint('Kept on Google Drive in', keep)"
    );
    let download = format!("shutil.copy('{stem_q}.szi', keep)\nfiles.download('{stem_q}.szi')");
    // Colab has PyTorch already (its CUDA build): only the rest is installed.
    let rest: Vec<&str> = requirements.split_whitespace().filter(|r| *r != "torch").collect();
    let mut setup = vec!["!nvidia-smi -L".to_owned()];
    if !rest.is_empty() {
        setup.push(format!("!pip install -q {}", rest.join(" ")));
    }
    let notebook = serde_json::json!({
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": { "provenance": [] },
            "kernelspec": { "name": "python3", "display_name": "Python 3" },
            "language_info": { "name": "python" }
        },
        "cells": [
            cell("markdown", &[
                &format!("# Training {stem} on Colab"),
                "",
                "1. Runtime › Change runtime type › a **GPU**.",
                &format!("2. Zip the folder *{folder}* on your computer."),
                "3. Run the cells in order. The first asks for the zip, the first time only.",
                "",
                &format!("Everything is kept on Google Drive, in *MyDrive/MacroMagi/{stem}*: the project, the run (saved \
                 every 20 minutes) and the corpora, cut into tokens once. When a session ends, run the cells again: it \
                 goes on where it stopped, with no upload. For a new export of the folder, set `NEW_ZIP = True` once."),
            ]),
            cell("code", &project.lines().collect::<Vec<_>>()),
            cell("code", &setup.iter().map(String::as_str).collect::<Vec<_>>()),
            cell("code", &["!python run.py"]),
            cell("code", &download.lines().collect::<Vec<_>>()),
        ]
    });
    serde_json::to_string_pretty(&notebook).unwrap_or_default()
}

fn copy_dir(from: &Path, to: &Path) -> Result<(), String> {
    std::fs::create_dir_all(to).map_err(|e| e.to_string())?;
    for entry in std::fs::read_dir(from).map_err(|e| format!("{}: {e}", from.display()))?.flatten() {
        let path = entry.path();
        let target = to.join(entry.file_name());
        if path.is_dir() {
            if entry.file_name() != "__pycache__" {
                copy_dir(&path, &target)?;
            }
        } else {
            std::fs::copy(&path, &target).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

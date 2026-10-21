//! The Magic tab's side of things that is not drawing: importing models, the
//! model being edited, and trying it out live.

pub mod export_python;
pub mod generate;
pub mod host_tools;
pub mod import;
pub mod train;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use magi_core::geom::Displays;
use magi_core::graph::{Fragment, Node, PortRef, Wire};
use magi_core::host::Host;
use magi_core::input::InputEvent;
use magi_core::magic::{DataItem, Decision, MagicFile, MagicModel, ModelKind, TextSpec};
use magi_core::nodes::ai::AiWatchParams;
use magi_core::nodes::flow::StartParams;
use magi_core::nodes::notes::CommentParams;
use magi_core::nodes::{Registry, ids, ports, studio};
use magi_core::project::Project;

use crate::history::History;

/// What undo and redo bring back: the canvas (the setup is in its blocks)
/// and the pictures. Data files are only dropped when the model is saved,
/// so undoing a removal finds them again.
#[derive(Clone, PartialEq)]
pub struct Snapshot {
    graph: magi_core::graph::Graph,
    training: Vec<DataItem>,
    finetune: Vec<DataItem>,
}

/// The model open in the Magic tab.
pub struct MagicDoc {
    pub file: MagicFile,
    pub path: Option<PathBuf>,
    /// What the importer found and guessed.
    pub notes: Vec<String>,
    /// The model's shapes, read when it was loaded.
    pub shape: Option<magi_ml::Shape>,
    /// The Magic tab's canvas, kept in `file.model.studio` too. The model's
    /// settings are made from it (`studio::derive`).
    pub studio: Project,
    /// Bumped when the model, weights or data files change, which the
    /// description alone does not show.
    revision: u64,
    /// The description and revision as last saved, to tell whether there are changes.
    saved: Option<(MagicModel, u64)>,
    /// The description and revision as last put into the macro, where it is kept too.
    kept: Option<(MagicModel, u64)>,
    /// The model (`fingerprint`) the package `.szi` holds, to write it again only when it changed.
    packaged: Option<u64>,
    history: History<Snapshot>,
}

/// A quick fingerprint of a model's weights and ONNX model: their lengths and their first and last
/// bytes – enough to tell a model trained again, without reading a large one through.
fn fingerprint(file: &MagicFile) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for bytes in [&file.weights, &file.onnx] {
        let edge = bytes.len().min(1 << 16);
        (bytes.len(), &bytes[..edge], &bytes[bytes.len() - edge..]).hash(&mut hasher);
    }
    hasher.finish()
}

impl MagicDoc {
    pub fn new(mut file: MagicFile, path: Option<PathBuf>, notes: Vec<String>) -> Self {
        let shape = (!file.onnx.is_empty()).then(|| magi_ml::inspect(&file.onnx).ok()).flatten();
        // A model from before the canvas gets one, built from its setup.
        let canvas = file.model.studio.take().unwrap_or_else(|| {
            let datasets: Vec<(String, bool)> = file.model.datasets.iter().map(|f| (f.clone(), false)).collect();
            studio::default_canvas(&file.model, !file.weights.is_empty(), !file.onnx.is_empty(), &datasets)
        });
        let mut graph = magi_core::graph::Graph::from(canvas);
        studio::upgrade(&mut graph);
        let studio = Project { graph, ..Project::new() };
        let history = History::new(&Snapshot {
            graph: studio.graph.clone(),
            training: file.model.training.clone(),
            finetune: file.model.finetune.clone(),
        });
        // A workspace opened with its package holds the package's model.
        let packaged =
            file.model.package.as_ref().filter(|p| std::path::Path::new(p).is_file()).map(|_| fingerprint(&file));
        let mut doc =
            Self { file, path, notes, shape, studio, revision: 0, saved: None, kept: None, packaged, history };
        doc.sync_canvas();
        doc.history = History::new(&doc.snapshot());
        doc.saved = doc.path.as_ref().map(|_| (doc.file.model.clone(), 0));
        // An ONNX copy of a text model the native engine runs from its weights is not kept – it
        // doubled the memory a trained model takes – unless Compile Package saves one.
        let native = doc.file.model.architecture.as_ref().is_some_and(|a| a.native_problem().is_none());
        if doc.file.model.kind == ModelKind::Text
            && native
            && !doc.file.weights.is_empty()
            && !doc.file.onnx.is_empty()
            && !doc.compile_settings().is_some_and(|c| c.onnx)
        {
            doc.file.onnx = Vec::new();
            doc.shape = None;
        }
        doc
    }

    /// Makes the model what the canvas says – its setup from the setup blocks
    /// chained into the start, its datasets from the Text Dataset blocks –
    /// notes what is not connected, and puts the canvas into the model.
    pub fn sync_canvas(&mut self) {
        let has_onnx = !self.file.onnx.is_empty();
        studio::derive(&mut self.studio.graph, &mut self.file.model, has_onnx);
        // The sliders' values, on what they change.
        let configured = self.file.model.configured(&serde_json::Map::new());
        if configured != self.file.model {
            self.file.model = configured;
        }
        let mut datasets: Vec<String> = Vec::new();
        for node in self.studio.graph.nodes().filter(|n| n.kind == studio::kinds::TEXT_DATA) {
            let params: studio::TextDataParams = node.params_or_default();
            for file in params.positive.into_iter().chain(params.negative) {
                if self.file.data.contains_key(&file) && !datasets.contains(&file) {
                    datasets.push(file);
                }
            }
        }
        if self.file.model.datasets != datasets {
            self.file.model.datasets = datasets;
        }
        let canvas: Fragment = self.studio.graph.clone().into();
        if self.file.model.studio.as_ref() != Some(&canvas) {
            self.file.model.studio = Some(canvas);
        }
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            graph: self.studio.graph.clone(),
            training: self.file.model.training.clone(),
            finetune: self.file.model.finetune.clone(),
        }
    }

    /// Call once per frame; `settled` is false during a drag or while typing.
    pub fn track_history(&mut self, settled: bool) {
        let snapshot = self.snapshot();
        self.history.track(&snapshot, settled);
    }

    pub fn can_undo(&self) -> bool {
        self.history.can_undo(&self.snapshot())
    }

    pub fn can_redo(&self) -> bool {
        self.history.can_redo()
    }

    pub fn undo(&mut self) {
        let mut snapshot = self.snapshot();
        self.history.undo(&mut snapshot);
        self.restore(snapshot);
    }

    pub fn redo(&mut self) {
        let mut snapshot = self.snapshot();
        self.history.redo(&mut snapshot);
        self.restore(snapshot);
    }

    fn restore(&mut self, snapshot: Snapshot) {
        self.studio.graph = snapshot.graph;
        self.file.model.training = snapshot.training;
        self.file.model.finetune = snapshot.finetune;
        self.sync_canvas();
    }

    /// Notes a change to the model, its weights or its data files.
    pub fn files_changed(&mut self) {
        self.revision += 1;
    }

    /// Puts what training made into the model.
    pub fn apply_trained(&mut self, trained: train::Trained) {
        self.files_changed();
        self.file.onnx = trained.onnx;
        self.file.weights = trained.weights;
        if let Some(tokenizer) = trained.tokenizer {
            self.set_tokenizer(tokenizer);
        }
        let model = &mut self.file.model;
        model.kind = trained.kind;
        model.architecture = Some(trained.architecture);
        let mut record = trained.record;
        // What it was trained to read first, so a changed system prompt can be pointed out.
        record.system = model.text.as_ref().map(|t| t.system.clone());
        model.trained = Some(record);
        if let Some(input) = trained.input {
            model.input = input;
            // The Picture Preparation block follows what the model was trained with.
            let value = studio::section_of(studio::kinds::PREP, &self.file.model);
            for node in self.studio.graph.nodes_mut().filter(|n| n.kind == studio::kinds::PREP) {
                node.set_param("value", value.clone());
            }
        }
        self.shape = (!self.file.onnx.is_empty()).then(|| magi_ml::inspect(&self.file.onnx).ok()).flatten();
        self.sync_canvas();
    }

    /// The Compile block's settings: the one on the Model wire, or any.
    pub fn compile_settings(&self) -> Option<studio::CompileParams> {
        studio::pipeline(&self.studio.graph).ok().and_then(|p| p.compile).map(|(_, params)| params).or_else(|| {
            self.studio.graph.nodes().find(|n| n.kind == studio::kinds::COMPILE).map(|n| n.params_or_default())
        })
    }

    /// Why the canvas cannot train the model now, if it cannot.
    pub fn training_problem(&self) -> Option<String> {
        let pipeline = match studio::pipeline(&self.studio.graph) {
            Ok(pipeline) => pipeline,
            Err(problem) => return Some(problem),
        };
        if !pipeline.trains() {
            return Some("Add a training stage after the model's start (and its architecture) to train it.".into());
        }
        if pipeline.entry.kind == studio::kinds::BASE
            && pipeline.entry.params_or_default::<studio::BaseParams>().path.trim().is_empty()
            && self.file.weights.is_empty()
        {
            return Some(
                "This model has no weights to train further: use Start from Zero, or choose a base model file.".into(),
            );
        }
        match &pipeline.compile {
            None => {
                return Some("Put Compile Package at the end of the Model wire: it saves the trained model.".into());
            }
            Some((_, params)) if params.folder.trim().is_empty() => {
                return Some("Choose the folder Compile Package saves into (select the block).".into());
            }
            _ => {}
        }
        let pictures = self.file.model.training.len() + self.file.model.finetune.len();
        let files = studio::PlanFiles {
            dataset: &|file: &str| self.file.data.contains_key(file).then(|| file.to_owned()),
            weights: None,
            pictures: (pictures > 0).then(String::new),
            out_dir: String::new(),
            devices: Vec::new(),
        };
        studio::plan(&pipeline, &self.studio.graph, &self.file.model, &files).err()
    }

    /// Saves the package as the Compile block says, into a folder of its own:
    /// the `.magic` workspace, the Magi model (one `.szi` file) and an ONNX file
    /// when asked for. Returns what was written, and what was skipped and why.
    pub fn compile(&mut self) -> Result<(Vec<PathBuf>, Vec<String>), String> {
        let params = self.compile_settings().ok_or("Add a Compile Package block to the canvas.")?;
        if params.folder.trim().is_empty() {
            return Err("Choose the folder Compile Package saves into.".into());
        }
        if !(params.magic || params.szi || params.onnx) {
            return Err("Choose what Compile Package saves: the .magic workspace, the .szi model or ONNX.".into());
        }
        let title = self.title();
        let stem = file_stem(if params.file.trim().is_empty() { &title } else { params.file.trim() });
        // One folder for the whole project: the workspace and every format of the model.
        let project = PathBuf::from(params.folder.trim()).join(&stem);
        std::fs::create_dir_all(&project)
            .map_err(|e| format!("The folder {} could not be made: {e}", project.display()))?;
        let mut written = Vec::new();
        let skipped = Vec::new();
        let trained = !self.file.weights.is_empty() || !self.file.onnx.is_empty();
        if self.file.model.blocks.is_none() {
            self.file.model.blocks = Some(starter_blocks(&self.file.model, "", &Registry::builtin()));
        }

        // The Magi model: one .szi file with its weights and its native tools.
        if params.szi && trained {
            let szi = project.join(format!("{stem}.{}", magi_core::szi::EXTENSION));
            self.write_szi(&szi, params.onnx)?;
            written.push(szi);
        }
        // The workspace: canvas, setup, and the datasets and pictures if chosen – never the model,
        // which goes into the .szi beside it (written too when the .szi switch is off).
        if params.magic {
            let include = &mut self.file.model.include;
            include.datasets = params.datasets;
            (include.training, include.finetune) = (params.pictures, params.pictures);
            include.blocks = params.blocks;
            let path = project.join(format!("{stem}.magic"));
            if let Some(szi) = self.save_workspace(&path)? {
                written.push(szi);
            }
            written.insert(0, path);
        }

        if params.onnx && !self.file.onnx.is_empty() {
            let folder = project.join("onnx");
            std::fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
            let onnx = folder.join(format!("{stem}.onnx"));
            std::fs::write(&onnx, &self.file.onnx).map_err(|e| format!("Could not save {}: {e}", onnx.display()))?;
            written.push(onnx);
        }
        Ok((written, skipped))
    }

    /// Writes the trained model as a `.szi` (with its ONNX model when it needs it, or `with_onnx`),
    /// and makes it the workspace's package.
    fn write_szi(&mut self, szi: &std::path::Path, with_onnx: bool) -> Result<(), String> {
        let with_onnx = with_onnx || magi_core::szi::needs_onnx(&self.file);
        magi_core::szi::write(&self.file, szi, with_onnx)
            .map_err(|e| format!("Could not write the Magi model: {e}"))?;
        self.file.model.package = Some(szi.display().to_string());
        self.packaged = Some(fingerprint(&self.file));
        Ok(())
    }

    /// Saves the workspace (`.magic`): the canvas and the setup, and the datasets and pictures as
    /// its `include` says – never the model. A trained model is kept in a `.szi` of the same name
    /// beside it, written when it is not there yet or the model changed; the workspace names it.
    /// Returns the `.szi` when one was written.
    pub fn save_workspace(&mut self, path: &std::path::Path) -> Result<Option<PathBuf>, String> {
        let trained = !self.file.weights.is_empty() || !self.file.onnx.is_empty();
        let mut written = None;
        if trained {
            let szi = path.with_extension(magi_core::szi::EXTENSION);
            let current =
                self.file.model.package.as_ref().is_some_and(|p| std::path::Path::new(p) == szi && szi.is_file())
                    && self.packaged == Some(fingerprint(&self.file));
            if !current {
                self.write_szi(&szi, false)?;
                written = Some(szi);
            }
        }
        let include = &mut self.file.model.include;
        (include.model, include.weights) = (false, false);
        self.file.save(path).map_err(|e| format!("Could not save {}: {e}", path.display()))?;
        self.mark_saved(path.to_path_buf());
        Ok(written)
    }

    pub fn is_dirty(&self) -> bool {
        self.saved.as_ref().is_none_or(|(model, revision)| *model != self.file.model || *revision != self.revision)
    }

    /// Whether closing it would lose changes: ones neither saved to a file
    /// nor put into the macro.
    pub fn would_lose_changes(&self) -> bool {
        self.is_dirty()
            && self
                .kept
                .as_ref()
                .is_none_or(|(model, revision)| *model != self.file.model || *revision != self.revision)
    }

    /// Notes that the model as it is now is in the macro.
    pub fn mark_kept(&mut self) {
        self.kept = Some((self.file.model.clone(), self.revision));
    }

    pub fn mark_saved(&mut self, path: PathBuf) {
        self.path = Some(path);
        self.saved = Some((self.file.model.clone(), self.revision));
    }

    pub fn title(&self) -> String {
        if self.file.model.name.trim().is_empty() { "Untitled model".into() } else { self.file.model.name.clone() }
    }

    /// Replaces the model itself, keeping the configuration.
    pub fn replace_onnx(&mut self, onnx: Vec<u8>) -> Result<(), String> {
        let shape = magi_ml::inspect(&onnx).map_err(|e| e.0)?;
        self.file.model.input = shape.suggested(self.file.model.input.clone());
        let value = studio::section_of(studio::kinds::PREP, &self.file.model);
        for node in self.studio.graph.nodes_mut().filter(|n| n.kind == studio::kinds::PREP) {
            node.set_param("value", value.clone());
        }
        self.file.onnx = onnx;
        self.shape = Some(shape);
        self.files_changed();
        Ok(())
    }
}

/// Where a training starts, and which stages it runs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrainChoice {
    pub from: TrainFrom,
    /// The stages to run (their blocks); empty: all of them.
    pub stages: Vec<magi_core::graph::NodeId>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub enum TrainFrom {
    /// As the canvas's start says: new weights, or the base model's.
    #[default]
    Start,
    /// The weights saved after a stage (its block).
    Checkpoint(magi_core::graph::NodeId),
    /// The model as it was last trained.
    Trained,
}

/// The weights saved after a stage of the last training that ran it.
#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub stage: magi_core::graph::NodeId,
    pub path: PathBuf,
    pub saved: Option<std::time::SystemTime>,
    pub loss: Option<f64>,
}

impl MagicDoc {
    /// Learns a learned tokenizer's pieces from the datasets the canvas trains
    /// on (and keeps them in the Text block). Returns how many, or `None` when
    /// the model's tokens are bytes.
    pub fn learn_tokens(&mut self) -> Option<usize> {
        let text = self.file.model.text.as_ref()?;
        if !text.tokenizer.learned() {
            return None;
        }
        let mut files: Vec<String> = studio::stage_files(&self.studio.graph).into_iter().flat_map(|(_, f)| f).collect();
        files.sort();
        files.dedup();
        let mut texts: Vec<String> = Vec::new();
        for file in &files {
            let Some(bytes) = self.file.data.get(file) else { continue };
            let content = String::from_utf8_lossy(bytes);
            let jsonl = studio::is_jsonl(file);
            for sample in studio::split_samples(file, &content) {
                texts.extend(magi_core::dataset::texts(&sample, jsonl));
            }
        }
        let mut spec = text.clone();
        let count = spec.learn_tokens(texts.iter().map(String::as_str));
        self.set_tokenizer(spec.tokenizer);
        Some(count)
    }

    /// Whether training reads a corpus where it runs (a Hugging Face Dataset block): then learned
    /// tokens are learned there, from it, rather than here from the files alone.
    pub fn reads_corpus(&self) -> bool {
        studio::pipeline(&self.studio.graph).is_ok_and(|p| {
            p.stages
                .iter()
                .flat_map(|s| &s.data)
                .any(|(node, _)| self.studio.graph.node(*node).is_some_and(|n| n.kind == studio::kinds::CORPUS))
        })
    }

    /// Learned tokens to learn again where training runs (their pieces dropped, their size kept).
    pub fn forget_tokens(&mut self) {
        if let Some(text) =
            self.file.model.text.as_ref().filter(|t| t.tokenizer.learned() && !t.tokenizer.merges.is_empty())
        {
            self.set_tokenizer(magi_core::magic::TokenizerSpec::learned_of(text.tokenizer.size));
        }
    }

    /// The model's tokens, put into the Text block, where the model's text setup lives.
    fn set_tokenizer(&mut self, tokenizer: magi_core::magic::TokenizerSpec) {
        let value = {
            let mut model = self.file.model.clone();
            if let Some(text) = model.text.as_mut() {
                text.tokenizer = tokenizer;
            }
            studio::section_of(studio::kinds::TEXT, &model)
        };
        for node in self.studio.graph.nodes_mut().filter(|n| n.kind == studio::kinds::TEXT) {
            node.set_param("value", value.clone());
        }
        self.sync_canvas();
    }

    /// When the canvas starts from a base model file (Load Base Model), the model reads text in the
    /// base's tokens – its weights learned those pieces – and its architecture blocks must be the
    /// base's. Takes the tokens; says what it did, or why it cannot start from that base.
    pub fn take_base_tokens(&mut self) -> Result<Option<String>, String> {
        let Ok(pipeline) = studio::pipeline(&self.studio.graph) else { return Ok(None) };
        if pipeline.entry.kind != studio::kinds::BASE {
            return Ok(None);
        }
        let params: studio::BaseParams = pipeline.entry.params_or_default();
        if params.path.trim().is_empty() {
            return Ok(None);
        }
        let base = magi_core::szi::open(std::path::Path::new(params.path.trim()))
            .map_err(|e| format!("The base model could not be opened: {e}"))?;
        let (Some(theirs), Some(mine)) = (base.model.text.as_ref(), self.file.model.text.as_ref()) else {
            return Ok(None);
        };
        let mut note = None;
        if theirs.tokenizer != mine.tokenizer {
            self.set_tokenizer(theirs.tokenizer.clone());
            note = Some(if theirs.tokenizer.merges.is_empty() {
                "Uses the base model's tokens: bytes.".to_owned()
            } else {
                format!("Uses the base model's tokens: {} pieces of words.", theirs.tokenizer.merges.len())
            });
        }
        if let Some(architecture) = &base.model.architecture {
            let base_key = Self::architecture_key(architecture, base.model.text.as_ref());
            if Self::architecture_key(&pipeline.architecture, self.file.model.text.as_ref()) != base_key {
                return Err(format!(
                    "The architecture blocks are not the base model's ({} layers, {} wide; its tags and tools \
                     decide its vocabulary too). File › New text model from a base sets them up.",
                    architecture.layers.len(),
                    architecture.width
                ));
            }
        }
        Ok(note)
    }

    /// The project's own folder (Compile's *Save into*, then its name).
    pub fn project_dir(&self) -> Option<PathBuf> {
        let params = self.compile_settings()?;
        if params.folder.trim().is_empty() {
            return None;
        }
        let title = self.title();
        let stem = file_stem(if params.file.trim().is_empty() { &title } else { params.file.trim() });
        Some(PathBuf::from(params.folder.trim()).join(stem))
    }

    /// Where the weights after each stage are kept.
    pub fn checkpoint_dir(&self) -> Option<PathBuf> {
        self.project_dir().map(|dir| dir.join("checkpoints"))
    }

    /// What the weights' shapes depend on, as a short key: checkpoints made with
    /// another architecture do not fit.
    pub fn architecture_key(architecture: &magi_core::magic::Architecture, text: Option<&TextSpec>) -> String {
        let mut value = serde_json::to_value(architecture).unwrap_or_default();
        // The tokens decide the size of the embedding table.
        if let (Some(object), Some(text)) = (value.as_object_mut(), text) {
            object.insert("vocabulary".into(), serde_json::json!(text.vocab_size()));
            object.insert("pieces".into(), serde_json::json!(text.tokenizer.merges));
        }
        if let Some(object) = value.as_object_mut() {
            // Neither changes a weight's shape.
            object.remove("blocks");
            object.remove("rope_scaling");
        }
        // FNV-1a: stable across versions, unlike the standard hasher.
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in value.to_string().bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
        }
        format!("{hash:016x}")
    }

    /// The checkpoints that fit the canvas as it is, in the order of its stages.
    pub fn checkpoints(&self) -> Vec<Checkpoint> {
        let (Some(dir), Ok(pipeline)) = (self.checkpoint_dir(), studio::pipeline(&self.studio.graph)) else {
            return Vec::new();
        };
        let key = Self::architecture_key(&pipeline.architecture, self.file.model.text.as_ref());
        let mut found = Vec::new();
        for stage in &pipeline.stages {
            let path = dir.join(format!("{}.safetensors", stage.node));
            let Ok(meta) = std::fs::read(dir.join(format!("{}.json", stage.node))) else { continue };
            let Ok(meta) = serde_json::from_slice::<serde_json::Value>(&meta) else { continue };
            if !path.is_file() || meta.get("architecture").and_then(|a| a.as_str()) != Some(key.as_str()) {
                continue;
            }
            let saved = meta
                .get("saved")
                .and_then(serde_json::Value::as_f64)
                .map(|s| std::time::UNIX_EPOCH + std::time::Duration::from_secs_f64(s.max(0.0)));
            found.push(Checkpoint { stage: stage.node, path, saved, loss: meta.get("loss").and_then(|l| l.as_f64()) });
        }
        found
    }

    /// The weights kept after a stage as a base model (`.szi`): the model's description with the
    /// canvas's architecture and tokens, those weights and its native tools – to fine-tune others from.
    pub fn save_base(&self, stage: magi_core::graph::NodeId, path: &std::path::Path, name: &str) -> Result<(), String> {
        let checkpoint = self
            .checkpoints()
            .into_iter()
            .find(|c| c.stage == stage)
            .ok_or("No weights are kept after that stage for this architecture: train it first.")?;
        let weights =
            std::fs::read(&checkpoint.path).map_err(|e| format!("The kept weights could not be read: {e}"))?;
        let pipeline = studio::pipeline(&self.studio.graph)?;
        let mut model = self.file.model.clone();
        model.name = name.to_owned();
        model.architecture = Some(pipeline.architecture.clone());
        let stages: Vec<String> = pipeline
            .stages
            .iter()
            .take_while(|s| s.node != stage)
            .chain(pipeline.stages.iter().filter(|s| s.node == stage))
            .map(|s| s.kind.trim_start_matches("magic.").to_owned())
            .collect();
        model.trained = Some(magi_core::magic::TrainingRecord {
            stages,
            loss: checkpoint.loss.unwrap_or(f64::NAN),
            parameters: self.file.model.trained.as_ref().map_or(0, |t| t.parameters),
            ..Default::default()
        });
        let file = MagicFile {
            model,
            weights,
            onnx: Vec::new(),
            data: self
                .file
                .data
                .iter()
                .filter(|(p, _)| p.starts_with("nt/"))
                .map(|(p, b)| (p.clone(), b.clone()))
                .collect(),
        };
        magi_core::szi::write(&file, path, false).map_err(|e| format!("Could not save {}: {e}", path.display()))
    }

    /// Whether the trained model fits the canvas's architecture (so it can be trained further).
    pub fn trained_fits(&self) -> bool {
        let Ok(pipeline) = studio::pipeline(&self.studio.graph) else { return false };
        !self.file.weights.is_empty()
            && self.file.model.architecture.as_ref().is_some_and(|a| {
                let text = self.file.model.text.as_ref();
                Self::architecture_key(a, text) == Self::architecture_key(&pipeline.architecture, text)
            })
    }

    /// A stage's name as people say it, with its own title when it has one.
    pub fn stage_title(&self, stage: magi_core::graph::NodeId) -> String {
        let Some(node) = self.studio.graph.node(stage) else { return "?".into() };
        if !node.label.trim().is_empty() {
            return node.label.clone();
        }
        studio::registry().get(&node.kind).map(|d| d.name.clone()).unwrap_or_default()
    }

    /// Training from a stage: from the weights saved after the stage before it
    /// (or from the start, for the first), through the stages after it.
    pub fn train_from_here(&self, stage: magi_core::graph::NodeId) -> Result<TrainChoice, String> {
        let pipeline = studio::pipeline(&self.studio.graph)?;
        let index =
            pipeline.stages.iter().position(|s| s.node == stage).ok_or("This stage is not on the Model wire.")?;
        let stages: Vec<_> = pipeline.stages[index..].iter().map(|s| s.node).collect();
        if index == 0 {
            return Ok(TrainChoice { from: TrainFrom::Start, stages });
        }
        let before = pipeline.stages[index - 1].node;
        if !self.checkpoints().iter().any(|c| c.stage == before) {
            return Err(format!(
                "No weights are saved after {} yet: train once with it, then start from here.",
                self.stage_title(before)
            ));
        }
        Ok(TrainChoice { from: TrainFrom::Checkpoint(before), stages })
    }

    /// The ways to train the canvas now, each with what it does: from the start,
    /// from after each stage that has saved weights, and each stage again on
    /// the trained model.
    pub fn train_choices(&self) -> Vec<(String, TrainChoice)> {
        let mut choices = vec![("Everything, from the start".to_owned(), TrainChoice::default())];
        let Ok(pipeline) = studio::pipeline(&self.studio.graph) else { return choices };
        let checkpoints = self.checkpoints();
        for index in 1..pipeline.stages.len() {
            let before = pipeline.stages[index - 1].node;
            if let Some(saved) = checkpoints.iter().find(|c| c.stage == before) {
                let rest: Vec<String> = pipeline.stages[index..].iter().map(|s| self.stage_title(s.node)).collect();
                choices.push((
                    format!("{} on, from after {} ({})", rest.join(" → "), self.stage_title(before), ago(saved.saved)),
                    TrainChoice {
                        from: TrainFrom::Checkpoint(before),
                        stages: pipeline.stages[index..].iter().map(|s| s.node).collect(),
                    },
                ));
            }
        }
        if self.trained_fits() {
            for stage in &pipeline.stages {
                choices.push((
                    format!("{} again, on the trained model", self.stage_title(stage.node)),
                    TrainChoice { from: TrainFrom::Trained, stages: vec![stage.node] },
                ));
            }
        }
        choices
    }
}

/// How long ago something was, in words.
pub fn ago(time: Option<std::time::SystemTime>) -> String {
    let Some(seconds) = time.and_then(|t| t.elapsed().ok()).map(|d| d.as_secs()) else {
        return "saved".into();
    };
    match seconds {
        0..60 => "saved just now".into(),
        60..3600 => format!("saved {} min ago", seconds / 60),
        3600..86_400 => format!("saved {} h ago", seconds / 3600),
        _ => format!("saved {} days ago", seconds / 86_400),
    }
}

/// A name usable as a file name.
pub fn file_stem(name: &str) -> String {
    let cleaned: String =
        name.chars().map(|c| if c.is_alphanumeric() || " -_.()".contains(c) { c } else { '-' }).collect();
    let cleaned = cleaned.trim().trim_matches('.').to_owned();
    if cleaned.is_empty() { "model".into() } else { cleaned }
}

/// A new model, with a canvas ready to train it: for a text model, Magi – an agent of about
/// 400M parameters with its tools and datasets, set up to train on a GPU server.
pub fn new_model(text: bool) -> MagicFile {
    new_model_of(text, &studio::MAGI)
}

/// A new model: Magi at `size` (text), or a small vision model.
pub fn new_model_of(text: bool, size: &studio::MagiSize) -> MagicFile {
    let mut model = MagicModel {
        name: if text { size.name } else { "My vision model" }.into(),
        description: if text {
            "An agent: it plans, looks up how things are done in MagiDB, and works through tools – code, files, documents, pictures and the web.".into()
        } else {
            "Tells the screen's states apart.".into()
        },
        ..MagicModel::default()
    };
    let mut data = std::collections::BTreeMap::new();
    let mut datasets = Vec::new();
    if text {
        model.kind = ModelKind::Text;
        // Magi's native tools, as its datasets use them.
        let mut spec = magi_core::magic::TextSpec {
            temperature: 0.3,
            // Room for a file written whole, or a long answer.
            max_tokens: 2048,
            system: studio::EXAMPLE_SYSTEM.into(),
            tools: studio::example_tools(),
            // It thinks before each step.
            thinking: Some(magi_core::magic::ThinkingSpec { on_by_default: true }),
            effort: Some(studio::magi_effort()),
            // 32K tokens learned where it trains, from the start of its corpora.
            tokenizer: magi_core::magic::TokenizerSpec::learned_of(32_768),
            // Its tools a line each: a third of the tokens the JSON form takes, in every sample.
            tool_list: "short".into(),
            ..Default::default()
        };
        spec.ensure_specials();
        model.text = Some(spec);
        model.memory = Some(magi_core::magic::MemorySpec::default());
        for (file, content, negative) in studio::example_datasets() {
            data.insert(file.clone(), content.into_bytes());
            datasets.push((file, negative));
        }
    } else {
        model.kind = ModelKind::Vision;
        model.input.width = 64;
        model.input.height = 64;
        model.classes = vec![
            magi_core::magic::ClassSpec { label: "Yes".into(), act: true, threshold: 0.8, ..Default::default() },
            magi_core::magic::ClassSpec { label: "No".into(), threshold: 0.5, ..Default::default() },
        ];
    }
    let canvas = studio::default_canvas_for(&model, false, false, &datasets, size);
    model.studio = Some(canvas);
    model.datasets = datasets.iter().map(|(f, _)| f.clone()).collect();
    MagicFile { model, data, ..MagicFile::default() }
}

/// Magi made tiny, on its own example files instead of the Hugging Face corpora: it trains in a
/// minute or two, for the tests that go the whole way.
#[cfg(test)]
pub fn tiny(doc: &mut MagicDoc) {
    use magi_core::graph::PortRef;
    let registry = studio::registry();
    let graph = &mut doc.studio.graph;
    let corpora: Vec<magi_core::graph::NodeId> =
        graph.nodes().filter(|n| n.kind == studio::kinds::CORPUS).map(|n| n.id).collect();
    // What each corpus fed is fed by the Text Dataset at the end of its With chain.
    let mut rewire = Vec::new();
    for corpus in &corpora {
        let mut source = graph.data_source(*corpus, "with").cloned();
        while let Some(found) = source.clone().filter(|s| corpora.contains(&s.node)) {
            source = graph.data_source(found.node, "with").cloned();
        }
        for wire in graph.wires().iter().filter(|w| w.from.node == *corpus && !corpora.contains(&w.to.node)) {
            if let Some(source) = &source {
                rewire.push((source.clone(), wire.to.clone()));
            }
        }
    }
    for corpus in corpora {
        graph.remove(corpus);
    }
    for (from, to) in rewire {
        graph.connect(&registry, from, PortRef::new(to.node, &to.port)).unwrap();
    }
    let ids: Vec<(magi_core::graph::NodeId, String)> = graph.nodes().map(|n| (n.id, n.kind.clone())).collect();
    for (id, kind) in ids {
        let node = graph.node_mut(id).unwrap();
        match kind.as_str() {
            studio::kinds::SIZE => {
                node.set_param("width", 96);
                node.set_param("heads", 4);
                node.set_param("kv_heads", 2);
                node.set_param("context", 1024);
            }
            studio::kinds::LAYERS => node.set_param("layers", 2),
            studio::kinds::MOE => node.set_param("experts", 4),
            studio::kinds::PRETRAIN | studio::kinds::FINETUNE => {
                node.set_param("steps", 60);
                node.set_param("batch", 4);
            }
            studio::kinds::COMPACTION => node.set_param("steps", 10),
            studio::kinds::RL => {
                node.set_param("steps", 4);
                node.set_param("group", 2);
                node.set_param("max_new", 32);
            }
            studio::kinds::OPTIMIZER => {
                node.set_param("warmup", 5);
                node.set_param("accumulate", 1);
            }
            _ => {}
        }
    }
    doc.set_tokenizer(magi_core::magic::TokenizerSpec::learned_of(1024));
    doc.sync_canvas();
}

/// One of the tools the AI Agent block carries out, as a tool of a model.
pub fn agent_tool(name: &str, about: &str) -> magi_core::magic::ToolSpec {
    let property = |ty: &str| serde_json::json!({ "type": ty });
    let properties = match name {
        "press" => serde_json::json!({ "key": property("string") }),
        "type" | "log" => serde_json::json!({ "text": property("string") }),
        "click" => {
            serde_json::json!({ "x": property("number"), "y": property("number"), "button": property("string") })
        }
        "move" => serde_json::json!({ "x": property("number"), "y": property("number") }),
        "wait" => serde_json::json!({ "ms": property("number") }),
        _ => serde_json::json!({}),
    };
    magi_core::magic::ToolSpec {
        name: name.to_owned(),
        description: about.split(" – ").nth(1).unwrap_or(about).to_owned(),
        parameters: serde_json::json!({ "type": "object", "properties": properties }),
        // The AI Agent block carries these out, not the model's native tools.
        native: false,
    }
}

// The blocks a model comes with: a Start block and an AI Watch block (an AI
/// Agent and a Log block for a text model), with a comment saying how to use
/// them. `model` is the asset id in the macro.
pub fn starter_blocks(model: &MagicModel, asset: &str, registry: &Registry) -> Fragment {
    let defaults = |kind: &str| registry.get(kind).map(|def| def.default_params()).unwrap_or_default();
    if model.kind == ModelKind::Text {
        return text_starter_blocks(model, asset, registry);
    }
    let start = Node::new(1, ids::START, [0.0, 0.0], {
        let mut params = defaults(ids::START);
        params.extend(
            serde_json::to_value(StartParams::default()).ok().and_then(|v| v.as_object().cloned()).unwrap_or_default(),
        );
        params
    });
    let watch_params =
        AiWatchParams { model: asset.to_owned(), display: model.capture.display, ..AiWatchParams::default() };
    let mut watch = Node::new(2, ids::AI_WATCH, [0.0, 130.0], defaults(ids::AI_WATCH));
    watch.set_params(&watch_params);
    let note = CommentParams {
        text: format!(
            "“{}” watches the screen while this runs and presses {} when it should. Press Run (or give the Start \
             block a hotkey) to start, and the stop hotkey to stop. Its settings are in the AI Watch block.",
            model.name,
            model.action.key.label()
        ),
        width: 260.0,
        height: 130.0,
        ..CommentParams::default()
    };
    let mut comment = Node::new(3, ids::COMMENT, [260.0, 0.0], defaults(ids::COMMENT));
    comment.label = model.name.clone();
    comment.set_params(&note);
    Fragment {
        nodes: vec![start, watch, comment],
        wires: vec![Wire { from: PortRef::new(1, ports::OUT), to: PortRef::new(2, ports::FLOW_IN) }],
        groups: Vec::new(),
    }
}

fn text_starter_blocks(model: &MagicModel, asset: &str, registry: &Registry) -> Fragment {
    let defaults = |kind: &str| registry.get(kind).map(|def| def.default_params()).unwrap_or_default();
    let mut start = Node::new(1, ids::START, [0.0, 0.0], defaults(ids::START));
    start.set_params(&StartParams::default());
    let which = magi_core::nodes::ai::AiModelParams { model: asset.to_owned() };
    let mut load = Node::new(2, ids::AI_LOAD, [0.0, 130.0], defaults(ids::AI_LOAD));
    load.set_params(&which);
    let mut agent = Node::new(3, ids::AI_AGENT, [0.0, 260.0], defaults(ids::AI_AGENT));
    agent.set_params(&magi_core::nodes::ai::AiAgentParams {
        model: asset.to_owned(),
        request: "Say hello".into(),
        ..Default::default()
    });
    let log = Node::new(4, ids::LOG, [0.0, 460.0], defaults(ids::LOG));
    let mut unload = Node::new(5, ids::AI_UNLOAD, [0.0, 590.0], defaults(ids::AI_UNLOAD));
    unload.set_params(&which);
    let note = CommentParams {
        text: format!(
            "AI Load Model loads “{}”, the AI Agent block asks it the request, the Log block writes its answer \
             and AI Unload Model frees its memory (a stopped macro unloads its models anyway). Turn on “Carry out \
             its tool calls” to let it press, type and click; its other tools come out of “Tool” and “Arguments” \
             for the blocks after it.",
            model.name
        ),
        width: 280.0,
        height: 170.0,
        ..CommentParams::default()
    };
    let mut comment = Node::new(6, ids::COMMENT, [280.0, 0.0], defaults(ids::COMMENT));
    comment.label = model.name.clone();
    comment.set_params(&note);
    Fragment {
        nodes: vec![start, load, agent, log, unload, comment],
        wires: vec![
            Wire { from: PortRef::new(1, ports::OUT), to: PortRef::new(2, ports::FLOW_IN) },
            Wire { from: PortRef::new(2, ports::OUT), to: PortRef::new(3, ports::FLOW_IN) },
            Wire { from: PortRef::new(3, ports::OUT), to: PortRef::new(4, ports::FLOW_IN) },
            Wire { from: PortRef::new(3, "answer"), to: PortRef::new(4, "value") },
            Wire { from: PortRef::new(4, ports::OUT), to: PortRef::new(5, ports::FLOW_IN) },
        ],
        groups: Vec::new(),
    }
}

/// What the live test shows.
#[derive(Default)]
pub struct TestState {
    /// The picture the model saw last (RGBA, as captured) and its size.
    pub picture: Option<(Vec<u8>, u32, u32)>,
    /// Bumped with every new picture, so the screen knows to update.
    pub frame: u64,
    pub probabilities: Vec<f32>,
    pub decision: Option<Decision>,
    pub looks_per_second: f64,
    pub engine: String,
    pub acted: u64,
    pub error: Option<String>,
}

/// Runs a model on the screen, over and over, in the background.
pub struct Tester {
    pub state: Arc<Mutex<TestState>>,
    /// The configuration in use; changes made while testing take effect at once.
    pub config: Arc<Mutex<MagicModel>>,
    stop: Arc<AtomicBool>,
}

impl Tester {
    /// Starts testing. With `press`, the model's key is pressed (through
    /// `host`) whenever the model says to act.
    pub fn start(
        onnx: Vec<u8>,
        config: MagicModel,
        displays: Displays,
        host: Arc<dyn Host>,
        press: bool,
        repaint: impl Fn() + Send + 'static,
    ) -> Self {
        let state = Arc::new(Mutex::new(TestState::default()));
        let config = Arc::new(Mutex::new(config));
        let stop = Arc::new(AtomicBool::new(false));
        let (for_state, for_config, for_stop) = (Arc::clone(&state), Arc::clone(&config), Arc::clone(&stop));
        std::thread::Builder::new()
            .name("magi-magic-test".into())
            .spawn(move || {
                run_test(&onnx, &for_config, &displays, host.as_ref(), press, &for_state, &for_stop, &repaint)
            })
            .expect("a thread can be started");
        Self { state, config, stop }
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl Drop for Tester {
    fn drop(&mut self) {
        self.stop();
    }
}

#[allow(clippy::too_many_arguments)]
fn run_test(
    onnx: &[u8],
    config: &Mutex<MagicModel>,
    displays: &Displays,
    host: &dyn Host,
    press: bool,
    state: &Mutex<TestState>,
    stop: &AtomicBool,
    repaint: &dyn Fn(),
) {
    let lock = || state.lock().unwrap_or_else(|p| p.into_inner());
    let spec = config.lock().unwrap_or_else(|p| p.into_inner()).input.clone();
    let mut model = match magi_ml::Model::load(onnx, &spec) {
        Ok(model) => model,
        Err(error) => {
            lock().error = Some(error.0);
            repaint();
            return;
        }
    };
    lock().engine = model.engine().label();
    let (mut looks, mut counted_from, mut last_repaint) = (0u32, Instant::now(), Instant::now());
    let mut resting_until: Option<Instant> = None;
    while !stop.load(Ordering::Relaxed) {
        let current = config.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let Some(rect) = current.capture.rect(displays, None) else {
            lock().error = Some("there is no display to look at".into());
            break;
        };
        let frame = match host.capture(Some(rect)) {
            Ok(frame) => frame,
            Err(error) => {
                lock().error = Some(error.0);
                break;
            }
        };
        let probabilities = match model.classify(&frame.rgba, frame.width, frame.height) {
            Ok(p) => p,
            Err(error) => {
                lock().error = Some(error.0);
                break;
            }
        };
        let decision = current.decide(&probabilities);
        looks += 1;
        let resting = resting_until.is_some_and(|until| Instant::now() < until);
        let act = decision.act && !resting;
        if act {
            resting_until = Some(
                Instant::now()
                    + Duration::from_secs_f64((decision.delay_ms + current.action.cooldown_ms).max(0.0) / 1000.0),
            );
        }
        {
            let mut shown = lock();
            shown.probabilities = probabilities;
            shown.decision = Some(decision.clone());
            if act {
                shown.acted += 1;
            }
            if last_repaint.elapsed() >= Duration::from_millis(33) {
                shown.picture = Some((frame.rgba, frame.width, frame.height));
                shown.frame += 1;
            }
            if counted_from.elapsed() >= Duration::from_secs(1) {
                shown.looks_per_second = f64::from(looks) / counted_from.elapsed().as_secs_f64();
                (looks, counted_from) = (0, Instant::now());
            }
        }
        if act && press {
            std::thread::sleep(Duration::from_secs_f64(decision.delay_ms.max(0.0) / 1000.0));
            let (down, up) = match current.action.key {
                magi_core::input::Button::Key(key) => (InputEvent::KeyDown { key }, InputEvent::KeyUp { key }),
                magi_core::input::Button::Mouse(button) => {
                    (InputEvent::ButtonDown { button }, InputEvent::ButtonUp { button })
                }
                magi_core::input::Button::Pad(_) => continue,
            };
            let _ = host.send(&down);
            std::thread::sleep(Duration::from_secs_f64(current.action.hold_ms.max(0.0) / 1000.0));
            let _ = host.send(&up);
        }
        if last_repaint.elapsed() >= Duration::from_millis(33) {
            last_repaint = Instant::now();
            repaint();
        }
    }
    repaint();
}

#[cfg(test)]
mod tests {
    use magi_core::magic::{ClassSpec, InputSpec};

    use super::*;

    #[test]
    fn a_new_text_model_is_magi_set_up_for_a_gpu_server() {
        let doc = MagicDoc::new(new_model(true), None, Vec::new());
        assert_eq!(doc.title(), "Magi");
        assert!(doc.reads_corpus(), "FineWeb-Edu and the code corpus feed pretraining");
        let pipeline = studio::pipeline(&doc.studio.graph).unwrap();
        assert_eq!((pipeline.architecture.width, pipeline.architecture.layers.len()), (768, 32));
        let text = doc.file.model.text.clone().unwrap();
        assert!(text.tokenizer.learned() && text.tokenizer.merges.is_empty() && text.tokenizer.size == 32_768);
        let tools: Vec<&str> = text.tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(tools, studio::MAGI_TOOLS, "Magi's native tools, in the datasets' order");
        assert!(text.tools.iter().all(|t| t.native));
        assert!(text.thinking.as_ref().is_some_and(|t| t.on_by_default));
        let files = studio::PlanFiles {
            dataset: &|name: &str| Some(name.to_owned()),
            weights: None,
            pictures: None,
            out_dir: "out".into(),
            devices: Vec::new(),
        };
        let plan = studio::plan(&pipeline, &doc.studio.graph, &doc.file.model, &files).unwrap();
        let pretrain = &plan["stages"][0];
        assert_eq!((pretrain["steps"].as_u64(), pretrain["batch"].as_u64()), (Some(22_900), Some(16)));
        assert_eq!(pretrain["optimizer"]["accumulate"].as_u64(), Some(4));
        let corpus = plan["datasets"].as_object().unwrap().values().filter_map(|v| v.as_array()).flatten();
        assert!(corpus.clone().any(|f| f["corpus"]["repo"] == "HuggingFaceFW/fineweb-edu"));
        assert!(corpus.into_iter().any(|f| f["corpus"]["field"] == "messages"), "smol-smoltalk to fine-tune on");
        // The workspace as Python, for a server: it fetches the corpora there.
        if let Some(out) = std::env::var_os("MAGI_TEST_EXPORT_LARGE") {
            let runtime = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/python");
            let dir = crate::magic::export_python::export(&doc, &runtime, std::path::Path::new(&out), false).unwrap();
            let requirements = std::fs::read_to_string(dir.join("requirements.txt")).unwrap();
            assert!(requirements.contains("huggingface_hub") && requirements.contains("tokenizers"));
        }
    }

    /// Magi 150M is Magi's design and recipe at a third of the size: 512 wide, 24 layers, 750M tokens.
    /// With MAGI_TEST_EXPORT_SMALL, it is exported as Python there.
    #[test]
    fn magi_150m_is_magi_at_a_smaller_size() {
        let doc = MagicDoc::new(new_model_of(true, &studio::MAGI_TEST), None, Vec::new());
        assert_eq!(doc.title(), "Magi 150M");
        let pipeline = studio::pipeline(&doc.studio.graph).unwrap();
        let architecture = &pipeline.architecture;
        assert_eq!((architecture.width, architecture.heads, architecture.layers.len()), (512, 8, 24));
        let files = studio::PlanFiles {
            dataset: &|file: &str| Some(file.to_owned()),
            weights: None,
            pictures: None,
            out_dir: "out".into(),
            devices: Vec::new(),
        };
        let plan = studio::plan(&pipeline, &doc.studio.graph, &doc.file.model, &files).unwrap();
        let steps: Vec<u64> = plan["stages"].as_array().unwrap().iter().filter_map(|s| s["steps"].as_u64()).collect();
        assert_eq!(steps, [5_700, 1_000, 100, 100]);
        if let Some(out) = std::env::var_os("MAGI_TEST_EXPORT_SMALL") {
            let runtime = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/python");
            crate::magic::export_python::export(&doc, &runtime, std::path::Path::new(&out), false).unwrap();
        }
    }

    /// A Magi model trained elsewhere (MAGI_TEST_SZI) opens and answers on the native engine.
    #[test]
    #[ignore = "needs a trained .szi"]
    fn a_trained_szi_answers_on_the_native_engine() {
        let path = std::path::PathBuf::from(std::env::var_os("MAGI_TEST_SZI").expect("MAGI_TEST_SZI"));
        let file = magi_core::szi::open(&path).unwrap();
        let architecture = file.model.architecture.clone().expect("an architecture");
        let memory = file.model.memory.clone().unwrap_or_default();
        let dir = std::env::temp_dir().join(format!("magi-szi-answers-{}", std::process::id()));
        let mut model = magi_ml::TextModel::load_native(&architecture, &file.weights, &memory, &dir).unwrap();
        let spec = file.model.text.clone().expect("a text model");
        let answer = magi_ml::ask(&mut model, &spec, "Hello!", 16, 0.0, 1).unwrap();
        println!("{} parameters' answer: {:?}", file.weights.len() / 4, answer.raw);
        assert!(!answer.raw.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_model_comes_with_blocks_that_run_it() {
        let registry = Registry::builtin();
        let model = MagicModel {
            name: "Watcher".into(),
            classes: vec![ClassSpec { label: "go".into(), act: true, ..ClassSpec::default() }],
            input: InputSpec::default(),
            ..MagicModel::default()
        };
        let blocks = starter_blocks(&model, "mdl-1", &registry);
        let mut graph = magi_core::graph::Graph::new();
        let ids = graph.insert_fragment(&blocks, [0.0, 0.0]);
        assert_eq!(ids.len(), 3);
        let watch = graph.nodes().find(|n| n.kind == ids::AI_WATCH).unwrap();
        assert_eq!(watch.param::<String>("model").as_deref(), Some("mdl-1"));
        let watch = watch.id;
        assert_eq!(graph.prune(&registry), 0, "the wire fits");
        let start = graph.nodes().find(|n| n.kind == ids::START).unwrap().id;
        assert_eq!(graph.flow_target(start, ports::OUT).map(|t| t.node), Some(watch));
    }
}

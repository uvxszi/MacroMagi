//! The blocks of the Magic tab: a canvas for making, training, setting up
//! and trying AI models. They never run in a macro; the Magic tab reads the
//! canvas.
//!
//! Four kinds of wire connect them:
//!
//! - **Model** – the model itself, left to right: a start, the architecture
//!   blocks in order, the training stages in order, the package, and the
//!   blocks that try the result.
//! - **Setup** – the model's own settings (its name, tools, answers…): setup
//!   blocks chain into each other and into the start's *Setup* input. What a
//!   block on that chain says is what the model is.
//! - **Feed-forward** – a Mixture of Experts or Persona Router block, wired
//!   into the layer blocks that use it instead of a plain feed-forward layer.
//! - **Data** and **Optimizer** – datasets and optimizer settings, wired into
//!   the training stages that use them.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::{Category, DataType, NodeDef, ParamDef, ParamKind, Registry};
use crate::graph::{Fragment, Graph, Group, Node, NodeId, PortRef, Wire};
use crate::magic::{
    ActionSpec, ArchBlock, Architecture, CaptureSpec, ClassSpec, EffortSpec, InputSpec, MagicModel, MemorySpec,
    ModelKind, ModelSetting, Persona, RopeScaling, TextSpec, ThinkingSpec, ToolSpec,
};

/// Block types of the Magic tab.
pub mod kinds {
    pub const ZERO: &str = "magic.zero";
    pub const BASE: &str = "magic.base";
    pub const CONFIG: &str = "magic.config";

    pub const INFO: &str = "magic.info";
    pub const TEXT: &str = "magic.text";
    pub const TOOLS: &str = "magic.tools";
    pub const NATIVE_TOOLS: &str = "magic.native_tools";
    pub const THINKING: &str = "magic.thinking";
    pub const EFFORT: &str = "magic.effort";
    pub const INPUT: &str = "magic.input";
    pub const ANSWERS: &str = "magic.answers";
    pub const ACTION: &str = "magic.action";
    pub const SETTINGS: &str = "magic.settings";
    pub const PREP: &str = "magic.prep";

    pub const SIZE: &str = "magic.size";
    pub const CONTEXT: &str = "magic.context_scaler";
    pub const MEMORY: &str = "magic.memory";
    pub const LAYERS: &str = "magic.layers";
    pub const WINDOW: &str = "magic.window";
    pub const MLA: &str = "magic.mla";
    pub const CSA: &str = "magic.csa";
    pub const HCA: &str = "magic.hca";
    pub const SSM: &str = "magic.ssm";
    pub const MOE: &str = "magic.moe";
    pub const PERSONA: &str = "magic.persona";
    pub const HYPER: &str = "magic.hyper";
    pub const MTP: &str = "magic.mtp";
    pub const SPLIT: &str = "magic.prefix_split";
    pub const CONV: &str = "magic.conv";

    pub const TEXT_DATA: &str = "magic.text_data";
    pub const CORPUS: &str = "magic.corpus";
    pub const IMAGE_DATA: &str = "magic.image_data";

    pub const PRETRAIN: &str = "magic.pretrain";
    pub const FINETUNE: &str = "magic.finetune";
    pub const RL: &str = "magic.rl";
    pub const COMPACTION: &str = "magic.compaction";
    pub const VISION_TRAIN: &str = "magic.vision_train";
    pub const OPTIMIZER: &str = "magic.optimizer";

    pub const COMPILE: &str = "magic.compile";
    pub const PROMPT: &str = "magic.prompt";
    pub const TEST: &str = "magic.test";

    /// Blocks that hold a part of the model's own settings, on the Setup wire.
    pub const SETUP_BLOCKS: [&str; 12] =
        [INFO, TEXT, TOOLS, NATIVE_TOOLS, THINKING, EFFORT, MEMORY, INPUT, ANSWERS, ACTION, SETTINGS, PREP];
    /// Setup blocks that switch something on: without one, it is off.
    pub const SWITCHES: [&str; 3] = [THINKING, EFFORT, MEMORY];
    /// Blocks that start a pipeline.
    pub const STARTS: [&str; 3] = [ZERO, BASE, CONFIG];
    /// Blocks that add layers (and take a feed-forward block).
    pub const LAYER_BLOCKS: [&str; 6] = [LAYERS, WINDOW, MLA, CSA, HCA, SSM];
    /// Architecture blocks on the model wire.
    pub const ARCHITECTURE: [&str; 12] = [SIZE, CONTEXT, LAYERS, WINDOW, MLA, CSA, HCA, SSM, HYPER, MTP, SPLIT, CONV];
    /// Training stages.
    pub const STAGES: [&str; 5] = [PRETRAIN, FINETUNE, RL, COMPACTION, VISION_TRAIN];
}

/// The wire a model travels along.
pub const MODEL: &str = "model";
/// The wire a model's own settings travel along.
pub const SETUP: &str = "setup";
/// A feed-forward block's output, and the layer blocks' input for it.
pub const FFN: &str = "ffn";
/// An Optimizer block's output, and the training stages' input for it.
pub const OPTIMIZER: &str = "optimizer";

// ---- block settings -----------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ZeroParams {
    /// `text` or `vision`.
    pub kind: String,
}

impl Default for ZeroParams {
    fn default() -> Self {
        Self { kind: "text".into() }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BaseParams {
    /// Empty: this model's own weights. Otherwise a `.magic` file to start from.
    pub path: String,
}

/// A setup block: its part of the model's settings, as JSON.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SetupParams {
    pub value: Value,
    /// What the canvas says about the block (set by the Magic tab).
    pub note: String,
}

/// The name and description of a model, for the Model Info block.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct InfoSection {
    pub name: String,
    pub description: String,
    pub author: String,
    pub license: String,
    pub source: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SizeParams {
    pub width: u32,
    pub heads: u32,
    pub kv_heads: u32,
    /// Text: the longest text, in tokens. Vision: the pictures' side, in pixels.
    pub context: u32,
    /// `rope` or `learned`.
    pub positions: String,
    pub rope_theta: f64,
    /// The output head shares the embedding table.
    pub tied: bool,
    /// Embeddings are scaled by √width (Gemma).
    pub embed_scale: bool,
}

impl Default for SizeParams {
    fn default() -> Self {
        Self {
            width: 128,
            heads: 4,
            kv_heads: 2,
            context: 256,
            positions: "rope".into(),
            rope_theta: 10000.0,
            tied: false,
            embed_scale: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LayersParams {
    pub layers: u32,
    /// The plain feed-forward layer's width, times the model's width.
    pub mult: f64,
    /// Shows the finer options in the panel.
    pub more: bool,
    /// RMSNorm on each head's queries and keys (Qwen3, Gemma 3).
    pub qk_norm: bool,
    /// `swiglu` or `geglu`.
    pub activation: String,
    /// `pre` (before each part) or `sandwich` (before and after: Gemma, GLM).
    pub norm: String,
    /// Local layers look only at the last this-many tokens; 0: every layer sees all.
    pub window: u32,
    /// With a window: every this-many-th layer sees all (Gemma 3: 6).
    pub global_every: u32,
    /// The share of each head that rotary positions turn (GLM: 0.5).
    pub rotary: f64,
    /// Biases on queries, keys and values (GLM, Qwen2).
    pub bias: bool,
}

impl Default for LayersParams {
    fn default() -> Self {
        Self {
            layers: 4,
            mult: 2.75,
            more: false,
            qk_norm: false,
            activation: "swiglu".into(),
            norm: "pre".into(),
            window: 0,
            global_every: 6,
            rotary: 1.0,
            bias: false,
        }
    }
}

impl LayersParams {
    /// The attention of one layer, `index` counted over the whole model.
    pub fn attention(&self, index: usize) -> Value {
        let local =
            self.window > 0 && (self.global_every == 0 || !(index + 1).is_multiple_of(self.global_every as usize));
        let mut attention = json!({ "kind": "full" });
        if self.qk_norm {
            attention["qk_norm"] = json!(true);
        }
        if self.bias {
            attention["bias"] = json!(true);
        }
        if self.rotary < 1.0 {
            attention["rotary"] = json!(self.rotary);
        }
        if local {
            attention["window"] = json!(self.window);
        }
        attention
    }
}

/// The Context Scaler block: train at a short context, run at a long one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextParams {
    /// The context it trains at; 0: Model Size's.
    #[serde(deserialize_with = "number_or_text")]
    pub train: u32,
    /// The context it runs at.
    #[serde(deserialize_with = "number_or_text")]
    pub target: u32,
    /// `yarn`, `ntk` or `none` (attention that copes by itself: compressed attention).
    pub method: String,
}

impl Default for ContextParams {
    fn default() -> Self {
        Self { train: 0, target: 262_144, method: "yarn".into() }
    }
}

impl ContextParams {
    /// The scaling it gives a model trained at `trained` tokens.
    pub fn scaling(&self, trained: u32) -> RopeScaling {
        let original = if self.train > 0 { self.train } else { trained }.max(1);
        let target = self.target.max(original);
        RopeScaling {
            kind: self.method.clone(),
            factor: if self.method == "none" { 1.0 } else { f64::from(target) / f64::from(original) },
            original,
            target,
        }
    }
}

/// A number from a choice setting (kept as text) or a number setting.
fn number_or_text<'de, D: serde::Deserializer<'de>>(d: D) -> Result<u32, D::Error> {
    match Value::deserialize(d)? {
        Value::Number(n) => Ok(n.as_u64().unwrap_or(0) as u32),
        Value::String(s) => s.trim().parse().map_err(serde::de::Error::custom),
        _ => Ok(0),
    }
}

/// A number of tokens as people write it: 256K, 8192.
pub fn tokens_label(tokens: u32) -> String {
    if tokens >= 1024 && tokens.is_multiple_of(1024) { format!("{}K", tokens / 1024) } else { tokens.to_string() }
}

/// The Prompt Model block: how requests are asked.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PromptParams {
    /// The context window, in tokens (0: the model's own).
    pub context: u32,
    /// The longest answer, in tokens (0: the model's own).
    pub max_tokens: u32,
    /// Below 0: the model's own.
    pub temperature: f64,
    /// `model`, `on` or `off`.
    pub thinking: String,
    /// Empty: the model's default.
    pub effort: String,
    /// Run the model's native tools and give it their results, until it answers.
    pub native_tools: bool,
    /// How many native tool calls one request may make.
    pub max_steps: u32,
    pub note: String,
}

impl Default for PromptParams {
    fn default() -> Self {
        Self {
            context: 0,
            max_tokens: 0,
            temperature: -1.0,
            thinking: "model".into(),
            effort: String::new(),
            native_tools: true,
            max_steps: 100,
            note: String::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WindowParams {
    pub layers: u32,
    pub window: u32,
    pub mult: f64,
}

impl Default for WindowParams {
    fn default() -> Self {
        Self { layers: 1, window: 64, mult: 2.75 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MlaParams {
    pub layers: u32,
    pub kv_rank: u32,
    pub q_rank: u32,
    pub rope_dim: u32,
    pub mult: f64,
}

impl Default for MlaParams {
    fn default() -> Self {
        Self { layers: 2, kv_rank: 64, q_rank: 96, rope_dim: 16, mult: 2.75 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CsaParams {
    pub layers: u32,
    pub compress: u32,
    pub top_k: u32,
    pub window: u32,
    pub indexer_heads: u32,
    pub indexer_dim: u32,
    pub rope_dim: u32,
    pub mult: f64,
}

impl Default for CsaParams {
    fn default() -> Self {
        Self {
            layers: 1,
            compress: 4,
            top_k: 16,
            window: 32,
            indexer_heads: 4,
            indexer_dim: 32,
            rope_dim: 16,
            mult: 2.75,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HcaParams {
    pub layers: u32,
    pub compress: u32,
    pub window: u32,
    pub rope_dim: u32,
    pub mult: f64,
}

impl Default for HcaParams {
    fn default() -> Self {
        Self { layers: 1, compress: 32, window: 32, rope_dim: 16, mult: 2.75 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SsmParams {
    pub layers: u32,
    pub state: u32,
    pub mult: f64,
}

impl Default for SsmParams {
    fn default() -> Self {
        Self { layers: 2, state: 16, mult: 2.75 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MoeParams {
    pub experts: u32,
    pub shared: u32,
    pub top_k: u32,
    /// Each expert's width, times the model's width.
    pub mult: f64,
    /// `softmax`, `sigmoid` or `sqrt_softplus`.
    pub gating: String,
    /// `aux` (a balance loss) or `bias` (auxiliary-loss-free).
    pub balance: String,
    /// Experts fixed per token by a hash of its id.
    pub hash: bool,
}

impl Default for MoeParams {
    fn default() -> Self {
        Self {
            experts: 8,
            shared: 1,
            top_k: 2,
            mult: 1.0,
            gating: "sigmoid".into(),
            balance: "bias".into(),
            hash: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PersonaParams {
    pub first: String,
    pub first_role: String,
    pub second: String,
    pub second_role: String,
    pub mult: f64,
}

impl Default for PersonaParams {
    fn default() -> Self {
        Self {
            first: "Gun".into(),
            first_role: "Strict code and tool execution".into(),
            second: "Magi".into(),
            second_role: "Breaking loops, critique, routing visual tools".into(),
            mult: 2.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HyperParams {
    pub streams: u32,
    pub sinkhorn: u32,
}

impl Default for HyperParams {
    fn default() -> Self {
        Self { streams: 4, sinkhorn: 20 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MtpParams {
    pub weight: f64,
}

impl Default for MtpParams {
    fn default() -> Self {
        Self { weight: 0.3 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ConvParams {
    pub layers: u32,
    pub channels: u32,
}

impl Default for ConvParams {
    fn default() -> Self {
        Self { layers: 3, channels: 32 }
    }
}

/// A Text Dataset block: files in the `.magic` file (`data/text/…`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TextDataParams {
    /// What to do: right answers, good tool calls, text to learn the language from.
    pub positive: Vec<String>,
    /// What not to do: answers to avoid, broken tool calls, loops.
    pub negative: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PretrainParams {
    pub steps: u32,
    pub batch: u32,
    /// 0: always the whole context. Otherwise the context grows from this many tokens.
    pub curriculum_start: u32,
}

impl Default for PretrainParams {
    fn default() -> Self {
        Self { steps: 400, batch: 16, curriculum_start: 0 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StageParams {
    pub steps: u32,
    pub batch: u32,
}

impl Default for StageParams {
    fn default() -> Self {
        Self { steps: 300, batch: 16 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FinetuneParams {
    /// `full`, `lora` or `dora`.
    pub method: String,
    pub rank: u32,
    pub alpha: f64,
    pub steps: u32,
    pub batch: u32,
    /// How strongly "what not to do" counts.
    pub negative_weight: f64,
}

impl Default for FinetuneParams {
    fn default() -> Self {
        Self { method: "full".into(), rank: 8, alpha: 16.0, steps: 600, batch: 16, negative_weight: 0.5 }
    }
}

/// One reward rule of reinforcement learning.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RewardRule {
    /// `contains`, `not_contains`, `regex`, `exact`, `tool_valid`,
    /// `tool_malformed`, `same_tool`, `unknown_tool`, `made_up_result`,
    /// `loop`, `short` or `ends`.
    pub kind: String,
    pub text: String,
    pub score: f64,
    /// `short`: at most this many tokens. `loop`: the length of the repeated run.
    pub tokens: u32,
    /// `loop`: how many repeats count as a loop.
    pub repeats: u32,
}

impl Default for RewardRule {
    fn default() -> Self {
        Self { kind: "contains".into(), text: String::new(), score: 1.0, tokens: 32, repeats: 3 }
    }
}

impl RewardRule {
    pub const KINDS: [(&'static str, &'static str); 12] = [
        ("contains", "Says…"),
        ("not_contains", "Does not say…"),
        ("regex", "Matches the pattern…"),
        ("exact", "Gives the dataset's answer"),
        ("tool_valid", "Calls tools correctly"),
        ("tool_malformed", "Writes a broken tool call"),
        ("same_tool", "Calls the tool the dataset's answer calls first"),
        ("unknown_tool", "Calls a tool it is not offered"),
        ("made_up_result", "Writes a tool's result itself"),
        ("loop", "Repeats itself (a loop)"),
        ("short", "Is short (fast)"),
        ("ends", "Finishes its answer"),
    ];
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RlParams {
    pub steps: u32,
    /// Answers drawn per request; each is scored against the others.
    pub group: u32,
    pub max_new: u32,
    pub temperature: f64,
    pub rules: Vec<RewardRule>,
}

impl Default for RlParams {
    fn default() -> Self {
        Self {
            steps: 60,
            group: 4,
            max_new: 64,
            temperature: 0.9,
            rules: vec![
                RewardRule { kind: "exact".into(), score: 1.0, ..RewardRule::default() },
                RewardRule { kind: "tool_valid".into(), score: 1.0, ..RewardRule::default() },
                RewardRule { kind: "short".into(), score: 5.0, tokens: 24, ..RewardRule::default() },
                RewardRule { kind: "loop".into(), score: -3.0, tokens: 4, ..RewardRule::default() },
                RewardRule { kind: "tool_malformed".into(), score: -1.0, ..RewardRule::default() },
            ],
        }
    }
}

/// A Hugging Face dataset read where training runs, too big to keep in the workspace.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CorpusParams {
    /// The dataset on huggingface.co, as `owner/name`.
    pub repo: String,
    /// Which of its parquet files, as a pattern.
    pub files: String,
    /// The column with the text.
    pub field: String,
    /// How many tokens of it to read, at most.
    pub tokens: u64,
}

impl Default for CorpusParams {
    fn default() -> Self {
        Self {
            repo: "HuggingFaceFW/fineweb-edu".into(),
            files: "sample/10BT/*.parquet".into(),
            field: "text".into(),
            tokens: 3_000_000_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OptimizerParams {
    /// `adamw`, `muon` or `sgd`.
    pub name: String,
    /// AdamW's or SGD's learning rate, for every weight.
    pub lr: f64,
    /// Muon's own learning rate, for the layers' weight matrices. Muon's updates are rescaled to
    /// AdamW's size, so its rate is of AdamW's order.
    pub muon_lr: f64,
    /// With Muon, AdamW's rate for the rest: embeddings (and the tied output head), norms, biases.
    pub embed_lr: f64,
    pub weight_decay: f64,
    /// `constant` or `cosine`.
    pub schedule: String,
    pub warmup: u32,
    /// Largest gradient norm; 0: no clipping.
    pub clip: f64,
    pub accumulate: u32,
    pub checkpointing: bool,
    pub offload: bool,
}

impl Default for OptimizerParams {
    fn default() -> Self {
        Self {
            // Tuned on the Magi presets (300 steps of pretraining, 12M: loss 2.25 at 0.02 and 0.002;
            // 1.93 here).
            name: "muon".into(),
            lr: 0.002,
            muon_lr: 0.004,
            embed_lr: 0.02,
            weight_decay: 0.01,
            schedule: "cosine".into(),
            warmup: 20,
            clip: 1.0,
            accumulate: 1,
            checkpointing: false,
            offload: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CompileParams {
    /// Where the package (and its exports) is saved after training.
    pub folder: String,
    /// The project's name (its folder and files); empty: the model's name.
    pub file: String,
    /// The `.magic` workspace: canvas and setup, to open and train again.
    pub magic: bool,
    pub weights: bool,
    pub datasets: bool,
    pub pictures: bool,
    pub blocks: bool,
    /// The Magi model: one `.szi` file with its weights and native tools.
    pub szi: bool,
    /// Also an ONNX file (`onnx/`), for ONNX runtimes.
    pub onnx: bool,
}

impl Default for CompileParams {
    fn default() -> Self {
        Self {
            folder: String::new(),
            file: String::new(),
            magic: true,
            weights: true,
            datasets: true,
            pictures: true,
            blocks: true,
            szi: true,
            onnx: false,
        }
    }
}

/// Settings of blocks with nothing of their own but a note.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NoteParams {
    pub note: String,
}

// ---- the registry -------------------------------------------------------------

fn model_in(def: NodeDef) -> NodeDef {
    def.data_in(MODEL, "Model", DataType::Any)
}

fn model_out(def: NodeDef) -> NodeDef {
    def.data_out(MODEL, "Model", DataType::Any)
}

fn chain(def: NodeDef) -> NodeDef {
    model_out(model_in(def))
}

fn int(min: i64, max: i64, unit: &str) -> ParamKind {
    ParamKind::int(min, max, unit)
}

fn float(min: f64, max: f64, unit: &str) -> ParamKind {
    ParamKind::float(min, max, unit)
}

fn note_param() -> ParamDef {
    ParamDef::new("note", "Note", ParamKind::Text { multiline: false }).hidden()
}

/// A block's face: its lines, then what the canvas says about it.
fn with_note(node: &Node, mut lines: Vec<String>) -> Vec<String> {
    if let Some(note) = node.param::<String>("note").filter(|n| !n.is_empty()) {
        lines.push(format!("⚠ {note}"));
    }
    lines
}

fn layers_line(layers: u32) -> String {
    if layers == 1 { "1 layer".into() } else { format!("{layers} layers") }
}

fn setup_block(registry: &mut Registry, id: &str, name: &str, description: &str, keywords: &[&str]) {
    registry.register(
        NodeDef::new(id, name, Category::ModelSetup, description)
            .keywords(keywords)
            .data_in(SETUP, "Setup", DataType::Any)
            .data_out(SETUP, "Setup", DataType::Any)
            .param(ParamDef::new("value", "Value", ParamKind::Text { multiline: true }).hidden())
            .param(note_param())
            .defaults(&SetupParams::default())
            .summarize(|node, _| {
                with_note(node, setup_lines(&node.kind, &node.params_or_default::<SetupParams>().value))
            }),
    );
}

/// The Magic tab's blocks.
pub fn registry() -> Registry {
    use kinds::*;
    let mut r = Registry::empty();

    // ---- where to start ----
    let start = |def: NodeDef| model_out(def.data_in(SETUP, "Setup", DataType::Any)).param(note_param());
    r.register(
        start(NodeDef::new(
            ZERO,
            "Start from Zero",
            Category::Start,
            "A brand-new model with random weights. The architecture blocks after it decide its shape; the \
             training stages after those teach it. Plug the setup blocks (Model Info, Text, Tools…) into Setup.",
        ))
        .keywords(&["new", "pretrain", "scratch", "zero", "start", "begin"])
        .param(ParamDef::new("kind", "Model", ParamKind::choice(&[("text", "Text / agent"), ("vision", "Vision")])))
        .defaults(&json!({ "kind": "text", "note": "" }))
        .summarize(|node, _| {
            let p: ZeroParams = node.params_or_default();
            with_note(node, vec![if p.kind == "vision" { "A new vision model" } else { "A new text model" }.into()])
        }),
    );
    r.register(
        start(NodeDef::new(
            BASE,
            "Load Base Model",
            Category::Start,
            "Starts from a trained model's weights, to train it further or fine-tune it: this model's own (leave \
             the path empty), or another .magic file trained in MacroMagi. The architecture blocks after it must \
             be the base model's.",
        ))
        .keywords(&["base", "load", "fine-tune", "adapt", "lora", "weights", "continue", "resume"])
        .param(
            ParamDef::new("path", "Base model", ParamKind::Path { folder: false })
                .help("Empty: this model's own weights."),
        )
        .defaults(&json!({ "path": "", "note": "" }))
        .summarize(|node, _| {
            let p: BaseParams = node.params_or_default();
            let line = if p.path.trim().is_empty() {
                "This model's weights".into()
            } else {
                p.path.rsplit(['/', '\\']).next().unwrap_or_default().to_owned()
            };
            with_note(node, vec![line])
        }),
    );
    r.register(
        start(NodeDef::new(
            CONFIG,
            "Use As It Is",
            Category::Start,
            "The model as it is – one imported from elsewhere, or trained before – without training: change its \
             setup, try it, package it.",
        ))
        .keywords(&["inference", "config", "configure", "try", "existing", "imported", "as is"])
        .defaults(&json!({ "note": "" }))
        .summarize(|node, _| with_note(node, vec!["The model as it is".into()])),
    );

    // ---- the model's setup ----
    setup_block(
        &mut r,
        INFO,
        "Model Info",
        "The model's name, what it does, who made it and its licence. Chain it into the start's Setup input.",
        &["info", "name", "author", "license", "licence", "description"],
    );
    setup_block(
        &mut r,
        TEXT,
        "Text",
        "Text models: what the model is told before every request (the system prompt), how freely it writes, how \
         long its answers may be, and its chat tags. Chain it into the start's Setup input.",
        &["prompt", "system", "chat", "temperature", "tokens", "tags"],
    );
    setup_block(
        &mut r,
        TOOLS,
        "Tools",
        "Text models: tools the model may call that the program running it carries out – in a macro the AI \
         Agent block (press, type, click…), elsewhere the app's own tools. Each has what it does and its \
         arguments; the model reads them in its system text. Chain it into the start's Setup input.",
        &["tools", "agent", "functions", "mcp", "calls"],
    );
    setup_block(
        &mut r,
        NATIVE_TOOLS,
        "Native Tools",
        "Text models: tools a Magi model runs itself, on this computer – a calculator, web search, the time, the \
         weather, files, pictures, sound. The model learns which to call and with what; the tool does the work and \
         its result comes back to the model. They go into the model's nt folder. Chain it into the start's Setup \
         input.",
        &["native tools", "tools", "magi", "nt", "calculator", "search", "agent"],
    );
    setup_block(
        &mut r,
        THINKING,
        "Thinking",
        "Text models: the model can be asked to think before answering, or not. Chained in, training teaches \
         both (reasoning in the datasets is learned with and without), the Prompt Model block gets a Thinking \
         switch. Without it, the model does not think: the datasets' \
         reasoning is left out.",
        &["thinking", "reasoning", "think", "toggle", "enable_thinking", "chain of thought"],
    );
    setup_block(
        &mut r,
        EFFORT,
        "Reasoning Effort",
        "Text models: levels of how hard the model reasons, such as standard and superior. Every request starts \
         by naming its level (\"Reasoning effort: superior\"), and the model learns each level from the samples \
         given it in the dataset editor. The Prompt Model block and the AI Agent window get a \
         Reasoning Effort choice.",
        &["effort", "reasoning", "levels", "low", "high", "reasoning_effort"],
    );
    setup_block(
        &mut r,
        MEMORY,
        "Memory & Cache",
        "Text models: where the model keeps the memory of what it has read while it runs. A hot tier (the first \
         block and the latest ones, at full precision), a warm tier in system memory (f16 or 8-bit), and a cold \
         tier in files on a disk (an NVMe SSD is best) – so long contexts never run out of memory. The system \
         text's memory can be kept on disk and found again by its SHA-256, so it is never read twice.",
        &["memory", "cache", "kv", "ssd", "offload", "context", "long", "quantized cache"],
    );
    setup_block(
        &mut r,
        INPUT,
        "Screen Area",
        "Vision models: the part of the screen the model looks at in a macro. Chain it into the start's Setup input.",
        &["screen", "area", "capture", "input", "region"],
    );
    setup_block(
        &mut r,
        ANSWERS,
        "Answers",
        "Vision models: the answers the model gives, whether to act on each, how sure it must be and how long to \
         wait. Training learns these answers. Chain it into the start's Setup input.",
        &["answers", "classes", "labels", "confidence", "threshold"],
    );
    setup_block(
        &mut r,
        ACTION,
        "When It Acts",
        "Vision models: the key or mouse button a macro presses when the model acts, how long, and how long \
         to ignore it afterwards. Chain it into the start's Setup input.",
        &["action", "key", "press", "cooldown"],
    );
    setup_block(
        &mut r,
        SETTINGS,
        "Sliders",
        "Vision models: sliders that change several values at once (how sure, how long to wait), offered again \
         in every macro block that uses the model. Chain it into the start's Setup input.",
        &["settings", "sliders", "adjustable"],
    );
    setup_block(
        &mut r,
        PREP,
        "Picture Preparation",
        "Vision models: how a picture is turned into numbers for the model – its size, layout, color order and \
         normalization. Importing and training set this; change it only if a model needs something else.",
        &["normalization", "mean", "std", "resize", "preprocess"],
    );

    // ---- architecture ----
    let ffn_in = |def: NodeDef| def.data_in(FFN, "Feed-forward", DataType::Any);
    r.register(
        chain(NodeDef::new(
            SIZE,
            "Model Size",
            Category::Architecture,
            "How big the model is: its width (numbers per token), attention heads, key/value heads (fewer is \
             smaller and faster), and its context – the longest text it reads, in tokens (or, for a vision model, \
             the pictures' side in pixels). Put it first, after the start.",
        ))
        .keywords(&["size", "width", "heads", "context", "dimensions", "hidden", "gqa", "rope"])
        .param(
            ParamDef::new("width", "Width", int(16, 8192, ""))
                .help("Numbers per token. Bigger learns more, and is slower."),
        )
        .param(ParamDef::new("heads", "Attention heads", int(1, 128, "")).help("The width must divide by it."))
        .param(
            ParamDef::new("kv_heads", "Key/value heads", int(1, 128, ""))
                .help("Heads must divide by it. Equal to the heads: plain attention."),
        )
        .param(ParamDef::new("context", "Context window", int(8, 32768, "")).help(
            "Text models: the longest text the model reads in training, in tokens (a piece of a word or a \
                     tag; 8,192 is about twenty pages). A Context Scaler stretches it when the model runs. Vision \
                     models: the side of the pictures, in pixels.",
        ))
        .param(
            ParamDef::new(
                "positions",
                "Positions",
                ParamKind::choice(&[("rope", "Rotary (RoPE)"), ("learned", "Learned")]),
            )
            .help("Rotary is what Llama and DeepSeek use, and what the Context Scaler stretches."),
        )
        .param(ParamDef::new("rope_theta", "Rotary base", float(100.0, 10_000_000.0, "")).show_if("positions", "rope"))
        .param(
            ParamDef::new("tied", "Tie input and output", ParamKind::Bool)
                .help("The output head reuses the embedding table: fewer weights (Qwen3 small models, Gemma)."),
        )
        .param(
            ParamDef::new("embed_scale", "Scale embeddings", ParamKind::Bool)
                .help("Embeddings are multiplied by √width before the first layer (Gemma)."),
        )
        .defaults(&SizeParams::default())
        .summarize(|node, _| {
            let p: SizeParams = node.params_or_default();
            vec![
                format!("{} wide · {} heads ({} kv)", p.width, p.heads, p.kv_heads),
                format!("context window {} tokens", p.context),
            ]
        }),
    );
    r.register(
        chain(NodeDef::new(
            CONTEXT,
            "Context Scaler",
            Category::Architecture,
            "Trains at a short context and runs at a long one: training stays fast and fits in a graphics card, \
             and the finished model reads up to the target. YaRN stretches the slow rotary channels and keeps the \
             fast ones, and sharpens attention so it stays focused over long texts; NTK raises the rotary base. \
             MacroMagi's native engine runs it. Put it after Model Size.",
        ))
        .keywords(&["context", "yarn", "rope", "ntk", "long", "extend", "scaling", "256k", "window"])
        .param(
            ParamDef::new(
                "train",
                "Train at",
                ParamKind::choice(&[("0", "Model Size's context"), ("2048", "2048 tokens"), ("4096", "4096 tokens"), ("8192", "8192 tokens")]),
            )
            .help("The context training uses. Short is fast and fits in 12 GB."),
        )
        .param(
            ParamDef::new(
                "target",
                "Run at",
                ParamKind::choice(&[("32768", "32K tokens"), ("65536", "64K tokens"), ("131072", "128K tokens"), ("262144", "256K tokens")]),
            )
            .help("The longest text the finished model reads."),
        )
        .param(ParamDef::new(
            "method",
            "Scaling",
            ParamKind::choice(&[("yarn", "YaRN"), ("ntk", "NTK (raise the rotary base)"), ("none", "None (compressed attention)")]),
        ).help("MacroMagi stretches positions only when a text is longer than the trained context, so short texts stay exactly as learned."))
        .param(note_param())
        .defaults(&json!({ "train": "0", "target": "262144", "method": "yarn", "note": "" }))
        .summarize(|node, _| {
            let p: ContextParams = node.params_or_default();
            let mut lines = vec![format!(
                "Trains at {}, runs at {}",
                if p.train == 0 { "Model Size's".to_owned() } else { tokens_label(p.train) },
                tokens_label(p.target)
            )];
            if p.train > 0 && p.method != "none" {
                let s = p.scaling(p.train);
                lines.push(format!("×{:.0} {} · attention ×{:.2}", s.factor, p.method.to_uppercase(), s.attention_factor()));
            }
            with_note(node, lines)
        }),
    );
    let mult = || {
        ParamDef::new("mult", "Feed-forward width", float(0.25, 16.0, "× width"))
            .help("The plain feed-forward layer, when no Mixture of Experts or Persona Router is wired in.")
    };
    r.register(
        ffn_in(chain(NodeDef::new(
            LAYERS,
            "Transformer Layers",
            Category::Architecture,
            "Standard layers (the Llama layout): every token looks at every earlier token (attention with rotary \
             positions), then a SwiGLU feed-forward layer – or the experts wired into Feed-forward.",
        )))
        .keywords(&["transformer", "attention", "layers", "llama", "backbone", "dense", "gqa"])
        .param(ParamDef::new("layers", "Layers", int(1, 256, "")))
        .param(mult())
        .param(
            ParamDef::new("more", "Finer options", ParamKind::Bool)
                .help("What sets Qwen, Gemma, GLM and Mistral apart from Llama."),
        )
        .param(
            ParamDef::new("qk_norm", "Normalize queries and keys", ParamKind::Bool)
                .help("RMSNorm on each head's queries and keys before they turn: steadier training (Qwen3, Gemma 3).")
                .show_if("more", true),
        )
        .param(
            ParamDef::new(
                "activation",
                "Feed-forward gate",
                ParamKind::choice(&[("swiglu", "SwiGLU (Llama, Qwen, Mistral)"), ("geglu", "GeGLU (Gemma)")]),
            )
            .show_if("more", true),
        )
        .param(
            ParamDef::new(
                "norm",
                "Normalization",
                ParamKind::choice(&[
                    ("pre", "Before each part (Llama, Qwen)"),
                    ("sandwich", "Before and after each part (Gemma, GLM)"),
                ]),
            )
            .show_if("more", true),
        )
        .param(
            ParamDef::new("window", "Local window", int(0, 32768, "tokens"))
                .help(
                    "0: every layer sees the whole text. Otherwise most layers look only at their own block of \
                     this many tokens and the block before – cheaper for long texts – and every few layers see \
                     everything.",
                )
                .show_if("more", true),
        )
        .param(
            ParamDef::new("global_every", "Every Nth layer sees all", int(0, 64, ""))
                .help("With a local window: Gemma 3 has five local layers, then a global one (6).")
                .show_if("more", true),
        )
        .param(
            ParamDef::new("rotary", "Rotary share", float(0.1, 1.0, ""))
                .help("How much of each head rotary positions turn: 1 for most models, 0.5 for GLM.")
                .show_if("more", true),
        )
        .param(
            ParamDef::new("bias", "Biases on queries, keys, values", ParamKind::Bool)
                .help("GLM and Qwen2 have them.")
                .show_if("more", true),
        )
        .param(note_param())
        .defaults(&json!({ "layers": 4, "mult": 2.75, "more": false, "qk_norm": false, "activation": "swiglu",
                           "norm": "pre", "window": 0, "global_every": 6, "rotary": 1.0, "bias": false, "note": "" }))
        .summarize(|node, _| {
            let p: LayersParams = node.params_or_default();
            let mut traits = Vec::new();
            if p.qk_norm {
                traits.push("QK-norm".to_owned());
            }
            if p.activation == "geglu" {
                traits.push("GeGLU".to_owned());
            }
            if p.norm == "sandwich" {
                traits.push("sandwich norms".to_owned());
            }
            if p.window > 0 {
                traits.push(format!("{}-token window", p.window));
            }
            if p.rotary < 1.0 {
                traits.push(format!("{:.0}% rotary", p.rotary * 100.0));
            }
            let mut lines = vec![layers_line(p.layers)];
            if !traits.is_empty() {
                lines.push(traits.join(" · "));
            }
            with_note(node, lines)
        }),
    );
    r.register(
        ffn_in(chain(NodeDef::new(
            WINDOW,
            "Sliding Window Attention",
            Category::Architecture,
            "Layers whose attention looks only at the last few tokens: cheap for long texts. DeepSeek-V4 starts \
             with these.",
        )))
        .keywords(&["sliding", "window", "local", "attention", "swa"])
        .param(ParamDef::new("layers", "Layers", int(1, 256, "")))
        .param(ParamDef::new("window", "Window", int(1, 32768, "tokens")))
        .param(mult())
        .param(note_param())
        .defaults(&json!({ "layers": 1, "window": 64, "mult": 2.75, "note": "" }))
        .summarize(|node, _| {
            let p: WindowParams = node.params_or_default();
            with_note(node, vec![layers_line(p.layers), format!("last {} tokens", p.window)])
        }),
    );
    r.register(
        ffn_in(chain(NodeDef::new(
            MLA,
            "Latent Attention (MLA)",
            Category::Architecture,
            "DeepSeek-V2/V3's multi-head latent attention: keys and values are rebuilt from a small latent per \
             token, so a model keeps far less in memory while it writes. A separate rotary part carries positions.",
        )))
        .keywords(&["mla", "latent", "attention", "deepseek", "v3", "kv", "cache", "compressed"])
        .param(ParamDef::new("layers", "Layers", int(1, 256, "")))
        .param(
            ParamDef::new("kv_rank", "Key/value latent", int(4, 4096, ""))
                .help("Numbers kept per token for keys and values."),
        )
        .param(ParamDef::new("q_rank", "Query latent", int(0, 4096, "")).help("0: queries straight from the token."))
        .param(ParamDef::new("rope_dim", "Rotary part", int(2, 512, "")).help("Channels that carry positions."))
        .param(mult())
        .param(note_param())
        .defaults(&json!({ "layers": 2, "kv_rank": 64, "q_rank": 96, "rope_dim": 16, "mult": 2.75, "note": "" }))
        .summarize(|node, _| {
            let p: MlaParams = node.params_or_default();
            with_note(node, vec![layers_line(p.layers), format!("latent {}", p.kv_rank)])
        }),
    );
    r.register(
        ffn_in(chain(NodeDef::new(
            CSA,
            "Compressed Sparse Attention",
            Category::Architecture,
            "DeepSeek-V4's CSA: every few tokens are pooled into one entry, a small “lightning indexer” picks the \
             entries that matter to each token, and the token attends to those plus a window of recent tokens. \
             Built for very long texts.",
        )))
        .keywords(&["csa", "sparse", "compressed", "attention", "deepseek", "v4", "indexer", "dsa"])
        .param(ParamDef::new("layers", "Layers", int(1, 256, "")))
        .param(ParamDef::new("compress", "Pool every", int(1, 1024, "tokens")))
        .param(
            ParamDef::new("top_k", "Entries per token", int(1, 4096, ""))
                .help("How many pooled entries the indexer picks for each token."),
        )
        .param(ParamDef::new("window", "Recent tokens", int(1, 32768, "")))
        .param(ParamDef::new("indexer_heads", "Indexer heads", int(1, 128, "")))
        .param(ParamDef::new("indexer_dim", "Indexer size", int(4, 1024, "")))
        .param(ParamDef::new("rope_dim", "Rotary part", int(2, 512, "")))
        .param(mult())
        .param(note_param())
        .defaults(
            &json!({ "layers": 1, "compress": 4, "top_k": 16, "window": 32, "indexer_heads": 4, "indexer_dim": 32,
                           "rope_dim": 16, "mult": 2.75, "note": "" }),
        )
        .summarize(|node, _| {
            let p: CsaParams = node.params_or_default();
            with_note(
                node,
                vec![layers_line(p.layers), format!("pool {} · top {} · window {}", p.compress, p.top_k, p.window)],
            )
        }),
    );
    r.register(
        ffn_in(chain(NodeDef::new(
            HCA,
            "Heavily Compressed Attention",
            Category::Architecture,
            "DeepSeek-V4's HCA: many tokens are pooled into each entry, and every token attends to all the pooled \
             entries before it plus a window of recent tokens. DeepSeek-V4 alternates it with Compressed Sparse \
             Attention.",
        )))
        .keywords(&["hca", "compressed", "attention", "deepseek", "v4", "pooled"])
        .param(ParamDef::new("layers", "Layers", int(1, 256, "")))
        .param(ParamDef::new("compress", "Pool every", int(1, 4096, "tokens")))
        .param(ParamDef::new("window", "Recent tokens", int(1, 32768, "")))
        .param(ParamDef::new("rope_dim", "Rotary part", int(2, 512, "")))
        .param(mult())
        .param(note_param())
        .defaults(&json!({ "layers": 1, "compress": 32, "window": 32, "rope_dim": 16, "mult": 2.75, "note": "" }))
        .summarize(|node, _| {
            let p: HcaParams = node.params_or_default();
            with_note(node, vec![layers_line(p.layers), format!("pool {} · window {}", p.compress, p.window)])
        }),
    );
    r.register(
        ffn_in(chain(NodeDef::new(
            SSM,
            "Mamba-2 / SSM",
            Category::Architecture,
            "State-space layers (Mamba-2): instead of attention, each token updates a small memory that fades over \
             time. Good at long, flowing text; often mixed with attention layers.",
        )))
        .keywords(&["mamba", "ssm", "state space", "recurrent", "linear"])
        .param(ParamDef::new("layers", "Layers", int(1, 256, "")))
        .param(ParamDef::new("state", "Memory size", int(2, 512, "")))
        .param(mult())
        .param(note_param())
        .defaults(&json!({ "layers": 2, "state": 16, "mult": 2.75, "note": "" }))
        .summarize(|node, _| {
            let p: SsmParams = node.params_or_default();
            with_note(node, vec![layers_line(p.layers), format!("memory {}", p.state)])
        }),
    );
    r.register(
        NodeDef::new(
            MOE,
            "Mixture of Experts",
            Category::Architecture,
            "DeepSeekMoE: replaces the feed-forward layer of the layer blocks it is wired into. Shared experts \
             see every token; each token also picks a few of many small routed experts, so the model holds a lot \
             but uses little per token. Wire its output into a layer block's Feed-forward input.",
        )
        .keywords(&["moe", "experts", "router", "mixture", "deepseek", "routed", "shared", "sparse"])
        .data_out(FFN, "Feed-forward", DataType::Any)
        .param(ParamDef::new("experts", "Routed experts", int(1, 1024, "")))
        .param(ParamDef::new("shared", "Shared experts", int(0, 16, "")))
        .param(ParamDef::new("top_k", "Experts per token", int(1, 64, "")))
        .param(ParamDef::new("mult", "Expert width", float(0.125, 16.0, "× width")))
        .param(
            ParamDef::new(
                "gating",
                "Choosing experts",
                ParamKind::choice(&[
                    ("softmax", "Softmax (V2)"),
                    ("sigmoid", "Sigmoid (V3)"),
                    ("sqrt_softplus", "√Softplus (V4)"),
                ]),
            )
            .help("How a token scores the experts. Sigmoid and √Softplus are normalized over the chosen ones."),
        )
        .param(
            ParamDef::new(
                "balance",
                "Balancing",
                ParamKind::choice(&[("bias", "Bias, no extra loss (V3/V4)"), ("aux", "Balance loss")]),
            )
            .help("Keeps tokens spread over the experts."),
        )
        .param(
            ParamDef::new("hash", "Fixed experts per token (hash)", ParamKind::Bool)
                .help("DeepSeek-V4's first layers: each token always uses the same experts."),
        )
        .param(note_param())
        .defaults(&json!({ "experts": 8, "shared": 1, "top_k": 2, "mult": 1.0, "gating": "sigmoid", "balance": "bias",
                           "hash": false, "note": "" }))
        .summarize(|node, _| {
            let p: MoeParams = node.params_or_default();
            let routing = if p.hash { " · hashed" } else { "" };
            with_note(
                node,
                vec![format!("{} routed + {} shared", p.experts, p.shared), format!("{} per token{routing}", p.top_k)],
            )
        }),
    );
    r.register(
        NodeDef::new(
            PERSONA,
            "Dual-Persona Router",
            Category::Architecture,
            "Two personas of the model's reasoning, each written inside its own tag and with its own feed-forward \
             expert: by default Gun (strict code and tool calls) and Magi (critique, breaking loops). Tool calls \
             keep one syntax. Wire its output into a layer block's Feed-forward input.",
        )
        .keywords(&["persona", "thinking", "reasoning", "gun", "magi", "router", "dual"])
        .data_out(FFN, "Feed-forward", DataType::Any)
        .param(ParamDef::new("first", "First persona", ParamKind::Text { multiline: false }))
        .param(ParamDef::new("first_role", "Its role", ParamKind::Text { multiline: false }))
        .param(ParamDef::new("second", "Second persona", ParamKind::Text { multiline: false }))
        .param(ParamDef::new("second_role", "Its role", ParamKind::Text { multiline: false }))
        .param(ParamDef::new("mult", "Expert width", float(0.25, 16.0, "× width")))
        .param(note_param())
        .defaults(&json!({ "first": "Gun", "first_role": "Strict code and tool execution", "second": "Magi",
                           "second_role": "Breaking loops, critique, routing visual tools", "mult": 2.0, "note": "" }))
        .summarize(|node, _| {
            let p: PersonaParams = node.params_or_default();
            with_note(node, vec![format!("<{}> & <{}>", tag_of(&p.first), tag_of(&p.second))])
        }),
    );
    r.register(
        chain(NodeDef::new(
            HYPER,
            "Hyper-Connections (mHC)",
            Category::Architecture,
            "DeepSeek-V4's manifold-constrained hyper-connections: the model carries several parallel streams \
             instead of one, mixed by learned, balanced weights around every layer. Steadier training for deep \
             models. Applies to the whole model.",
        ))
        .keywords(&["mhc", "hyper", "connections", "residual", "deepseek", "v4", "streams"])
        .param(ParamDef::new("streams", "Streams", int(2, 16, "")))
        .param(ParamDef::new("sinkhorn", "Balancing rounds", int(1, 100, "")))
        .defaults(&HyperParams::default())
        .summarize(|node, _| vec![format!("{} streams", node.params_or_default::<HyperParams>().streams)]),
    );
    r.register(
        chain(NodeDef::new(
            MTP,
            "Multi-Token Prediction",
            Category::Architecture,
            "While training, an extra small layer also predicts the token after next, which teaches the model \
             to plan ahead (DeepSeek-V3/V4). It is left out of the finished model.",
        ))
        .keywords(&["mtp", "multi-token", "prediction", "deepseek", "lookahead"])
        .param(ParamDef::new("weight", "Weight", float(0.0, 2.0, "")).help("How much the extra prediction counts."))
        .defaults(&MtpParams::default())
        .summarize(|node, _| vec![format!("weight {}", node.params_or_default::<MtpParams>().weight)]),
    );
    r.register(
        chain(NodeDef::new(
            SPLIT,
            "Encoder/Decoder Split",
            Category::Architecture,
            "The request (everything up to the answer) is read in both directions at once, and only the answer is \
             written token by token. Applies to the whole model.",
        ))
        .keywords(&["prefix", "encoder", "decoder", "split", "bidirectional"])
        .defaults(&Map::new())
        .summarize(|_, _| vec!["Request both ways".into()]),
    );
    r.register(
        chain(NodeDef::new(
            CONV,
            "Vision Encoder",
            Category::Architecture,
            "Vision models: convolution layers that turn a picture into features, then the answers.",
        ))
        .keywords(&["vision", "cnn", "convolution", "encoder", "image"])
        .param(ParamDef::new("layers", "Layers", int(1, 8, "")))
        .param(ParamDef::new("channels", "Channels", int(4, 512, "")))
        .defaults(&ConvParams::default())
        .summarize(|node, _| {
            let p: ConvParams = node.params_or_default();
            vec![format!("{} · {} channels", layers_line(p.layers), p.channels)]
        }),
    );

    // ---- data ----
    r.register(
        NodeDef::new(
            TEXT_DATA,
            "Text Dataset",
            Category::Data,
            "Text the model learns from, kept in the .magic file: .jsonl requests and answers, conversations and \
             tool calls, and .md or .txt text. “What to do” feeds Pretraining and Fine-Tuning; “What not to do” \
             feeds Fine-Tuning's input of the same name. Select it to add, write and check samples.",
        )
        .keywords(&["dataset", "text", "jsonl", "markdown", "samples", "data", "negative", "examples"])
        .data_out("positive", "What to do", DataType::Any)
        .data_out("negative", "What not to do", DataType::Any)
        .param(ParamDef::new("positive", "What to do", ParamKind::Text { multiline: false }).hidden())
        .param(ParamDef::new("negative", "What not to do", ParamKind::Text { multiline: false }).hidden())
        .param(note_param())
        .defaults(&json!({ "positive": [], "negative": [], "note": "" }))
        .summarize(|node, _| {
            let p: TextDataParams = node.params_or_default();
            let files = |n: usize| if n == 1 { "1 file".to_owned() } else { format!("{n} files") };
            with_note(
                node,
                vec![format!("To do: {}", files(p.positive.len())), format!("Not to do: {}", files(p.negative.len()))],
            )
        }),
    );
    r.register(
        NodeDef::new(
            CORPUS,
            "Hugging Face Dataset",
            Category::Data,
            "A big corpus of text on huggingface.co to pretrain on – billions of tokens, too many to keep in the \
             workspace. It is read where the training runs (best a GPU server: File › Export as Python), cut into \
             the model's tokens once and kept there; a model whose tokens are not learned yet learns them from its \
             start. FineWeb-Edu (educational web pages, ODC-By licence) is the default. Wire it into Pretraining; \
             a Text Dataset wired into its With input is read along with it.",
        )
        .keywords(&["hugging face", "corpus", "fineweb", "pretraining", "dataset", "web", "big", "stream", "parquet"])
        .data_in("with", "With", DataType::Any)
        .data_out("positive", "Text", DataType::Any)
        .param(
            ParamDef::new("repo", "Dataset", ParamKind::Text { multiline: false })
                .help("On huggingface.co, as owner/name."),
        )
        .param(ParamDef::new("files", "Files", ParamKind::Text { multiline: false }).help(
            "Which of its parquet files, as a pattern: sample/10BT/*.parquet is FineWeb-Edu's 10-billion-token sample.",
        ))
        .param(ParamDef::new("field", "Text column", ParamKind::Text { multiline: false }))
        .param(
            ParamDef::new("tokens", "Read at most", int(1_000_000, 1_000_000_000_000, "tokens"))
                .help("About 20 tokens a parameter is plenty; a few billion make a capable small model."),
        )
        .param(note_param())
        .defaults(&{
            let mut v = serde_json::to_value(CorpusParams::default()).expect("serializable");
            v["note"] = json!("");
            v
        })
        .summarize(|node, _| {
            let p: CorpusParams = node.params_or_default();
            with_note(node, vec![p.repo.clone(), format!("{:.1}B tokens", p.tokens as f64 / 1e9)])
        }),
    );
    r.register(
        NodeDef::new(
            IMAGE_DATA,
            "Vision Dataset",
            Category::Data,
            "Pictures the model learns from: pictures of each answer, and pictures an answer was mistaken for. \
             Wire it into Vision Training. Select it to add and remove pictures.",
        )
        .keywords(&["dataset", "pictures", "images", "vision", "positive", "negative"])
        .data_out("pictures", "Pictures", DataType::Any)
        .param(note_param())
        .defaults(&json!({ "note": "" }))
        .summarize(|node, _| with_note(node, vec!["The model's pictures".into()])),
    );

    // ---- training ----
    let stage = |def: NodeDef| chain(def).data_in(OPTIMIZER, "Optimizer", DataType::Any).param(note_param());
    r.register(
        stage(
            NodeDef::new(
                PRETRAIN,
                "Pretraining",
                Category::Training,
                "Teaches a new model the language: it reads all the text in its dataset and learns to predict each \
                 next token. Comes first for a model started from zero.",
            )
            .data_in("data", "Text", DataType::Any),
        )
        .keywords(&["pretrain", "pretraining", "language model", "tokens", "next token"])
        .param(ParamDef::new("steps", "Steps", int(1, 10_000_000, "")).help("How many batches to learn from."))
        .param(ParamDef::new("batch", "Batch", int(1, 4096, "")).help("Texts learned from at once."))
        .param(
            ParamDef::new("curriculum_start", "Start the context at", int(0, 32768, "tokens"))
                .help("0: always the whole context. Otherwise texts start this short and grow to the full context."),
        )
        .defaults(&json!({ "steps": 400, "batch": 16, "curriculum_start": 0, "note": "" }))
        .summarize(|node, _| {
            let p: PretrainParams = node.params_or_default();
            with_note(node, vec![format!("{} steps · batch {}", p.steps, p.batch)])
        }),
    );
    r.register(
        stage(
            NodeDef::new(
                FINETUNE,
                "Fine-Tuning",
                Category::Training,
                "Teaches the model to answer: the requests and answers in “What to do” become more likely, and the \
                 answers in “What not to do” less likely. Train all weights, or small LoRA or DoRA adapters that \
                 are folded back in afterwards.",
            )
            .data_in("data", "What to do", DataType::Any)
            .data_in("negative", "What not to do", DataType::Any),
        )
        .keywords(&["fine-tune", "finetune", "sft", "lora", "dora", "adapter", "instruct"])
        .param(ParamDef::new("method", "Train", ParamKind::choice(&[("full", "All weights"), ("lora", "LoRA adapters"), ("dora", "DoRA adapters")])))
        .param(ParamDef::new("rank", "Adapter rank", int(1, 512, "")).help("Bigger adapters learn more.").show_if("method", "lora"))
        .param(ParamDef::new("alpha", "Adapter strength", float(0.1, 512.0, "")).show_if("method", "lora"))
        .param(ParamDef::new("steps", "Steps", int(1, 10_000_000, "")))
        .param(ParamDef::new("batch", "Batch", int(1, 4096, "")))
        .param(ParamDef::new("negative_weight", "“Not to do” weight", float(0.0, 10.0, "")).help("How strongly answers to avoid count."))
        .defaults(&json!({ "method": "full", "rank": 8, "alpha": 16.0, "steps": 600, "batch": 16, "negative_weight": 0.5, "note": "" }))
        .summarize(|node, _| {
            let p: FinetuneParams = node.params_or_default();
            let method = match p.method.as_str() {
                "lora" => format!("LoRA rank {}", p.rank),
                "dora" => format!("DoRA rank {}", p.rank),
                _ => "All weights".into(),
            };
            with_note(node, vec![format!("{method} · {} steps", p.steps)])
        }),
    );
    r.register(
        stage(
            NodeDef::new(
                RL,
                "Reinforcement Learning",
                Category::Training,
                "The model answers each request several times; its reward rules score the answers (right answer, \
                 good tool calls, short, no loops…), and the better-than-average answers become more likely. Comes \
                 after fine-tuning.",
            )
            .data_in("data", "Requests", DataType::Any),
        )
        .keywords(&["rl", "reinforcement", "reward", "grpo", "score", "rlhf"])
        .param(ParamDef::new("steps", "Steps", int(1, 1_000_000, "")))
        .param(ParamDef::new("group", "Answers per request", int(2, 64, "")))
        .param(ParamDef::new("max_new", "Longest answer", int(1, 8192, "tokens")))
        .param(
            ParamDef::new("temperature", "Temperature", float(0.05, 2.0, "")).help("How varied the tried answers are."),
        )
        .param(ParamDef::new("rules", "Reward rules", ParamKind::Text { multiline: false }).hidden())
        .defaults(&{
            let mut v = serde_json::to_value(RlParams::default()).expect("serializable");
            v["note"] = json!("");
            v
        })
        .summarize(|node, _| {
            let p: RlParams = node.params_or_default();
            with_note(
                node,
                vec![
                    format!("{} steps · {} answers each", p.steps, p.group),
                    format!("{} reward rules", p.rules.len()),
                ],
            )
        }),
    );
    r.register(
        stage(
            NodeDef::new(
                COMPACTION,
                "Context Compaction",
                Category::Training,
                "Teaches the model to rewrite a long context – a conversation, a log – into a short one that keeps \
                 what matters, from {\"context\", \"compact\"} samples.",
            )
            .data_in("data", "Samples", DataType::Any),
        )
        .keywords(&["compaction", "compact", "summarize", "context", "memory"])
        .param(ParamDef::new("steps", "Steps", int(1, 10_000_000, "")))
        .param(ParamDef::new("batch", "Batch", int(1, 4096, "")))
        .defaults(&json!({ "steps": 300, "batch": 16, "note": "" }))
        .summarize(|node, _| {
            let p: StageParams = node.params_or_default();
            with_note(node, vec![format!("{} steps · batch {}", p.steps, p.batch)])
        }),
    );
    r.register(
        stage(
            NodeDef::new(
                VISION_TRAIN,
                "Vision Training",
                Category::Training,
                "Teaches a vision model its answers from the pictures of a Vision Dataset; pictures an answer was \
                 mistaken for make it less likely there.",
            )
            .data_in("pictures", "Pictures", DataType::Any),
        )
        .keywords(&["vision", "train", "classifier", "pictures"])
        .param(ParamDef::new("steps", "Steps", int(1, 10_000_000, "")))
        .param(ParamDef::new("batch", "Batch", int(1, 4096, "")))
        .defaults(&json!({ "steps": 300, "batch": 16, "note": "" }))
        .summarize(|node, _| {
            let p: StageParams = node.params_or_default();
            with_note(node, vec![format!("{} steps · batch {}", p.steps, p.batch)])
        }),
    );
    r.register(
        NodeDef::new(
            OPTIMIZER,
            "Optimizer",
            Category::Training,
            "How the training stages it is wired into learn: the optimizer (AdamW, or Muon as DeepSeek-V4 uses, \
             with AdamW for embeddings and norms), the learning rate and its schedule, and ways to save memory. \
             Wire its output into each stage's Optimizer input; one block can feed several stages.",
        )
        .keywords(&["optimizer", "muon", "adamw", "memory", "vram", "offload", "learning rate", "schedule", "warmup"])
        .data_out(OPTIMIZER, "Optimizer", DataType::Any)
        .param(ParamDef::new(
            "name",
            "Optimizer",
            ParamKind::choice(&[("adamw", "AdamW"), ("muon", "Muon (+ AdamW)"), ("sgd", "SGD")]),
        ))
        .param(
            ParamDef::new("lr", "Learning rate", float(0.000_001, 1.0, ""))
                .help("How big each step is. Too high: the loss jumps around; too low: it barely moves.")
                .show_unless("name", "muon"),
        )
        .param(
            ParamDef::new("muon_lr", "Learning rate (layers)", float(0.000_01, 1.0, ""))
                .help("Muon's, for the layers' weight matrices. Too high: the loss jumps around; too low: it barely moves.")
                .show_if("name", "muon"),
        )
        .param(
            ParamDef::new("embed_lr", "Learning rate (embeddings)", float(0.000_001, 1.0, ""))
                .help("AdamW's, for the rest: the embeddings (and the output head when tied), norms and biases.")
                .show_if("name", "muon"),
        )
        .param(ParamDef::new("weight_decay", "Weight decay", float(0.0, 1.0, "")))
        .param(ParamDef::new(
            "schedule",
            "Schedule",
            ParamKind::choice(&[("cosine", "Cosine decay"), ("constant", "Constant")]),
        ))
        .param(ParamDef::new("warmup", "Warm up over", int(0, 1_000_000, "steps")))
        .param(ParamDef::new("clip", "Clip gradients at", float(0.0, 100.0, "")).help("0: no clipping."))
        .param(
            ParamDef::new("accumulate", "Accumulate", int(1, 1024, "batches"))
                .help("Learns from several batches per step: a bigger batch in the same memory."),
        )
        .param(ParamDef::new("checkpointing", "Recompute activations (less memory)", ParamKind::Bool))
        .param(
            ParamDef::new("offload", "Optimizer state in system memory", ParamKind::Bool)
                .help("Less graphics memory, a little slower."),
        )
        .param(note_param())
        .defaults(&{
            let mut v = serde_json::to_value(OptimizerParams::default()).expect("serializable");
            v["note"] = json!("");
            v
        })
        .summarize(|node, _| {
            let p: OptimizerParams = node.params_or_default();
            let name = match p.name.as_str() {
                "muon" => "Muon",
                "sgd" => "SGD",
                _ => "AdamW",
            };
            let rate = if p.name == "muon" { format!("{} / {}", p.muon_lr, p.embed_lr) } else { p.lr.to_string() };
            with_note(node, vec![format!("{name} · {rate}"), format!("{} schedule", p.schedule)])
        }),
    );

    // ---- packaging ----
    r.register(
        chain(NodeDef::new(
            COMPILE,
            "Compile Package",
            Category::Package,
            "Saves the finished model after training into a folder of its own in the folder you choose: the \
             .magic workspace (canvas, setup, datasets) and the Magi model (.szi: its weights and native \
             tools), and an ONNX file if you like. Training needs a folder here.",
        ))
        .keywords(&["compile", "package", "export", "save", "magic", "szi", "onnx", "folder"])
        .param(ParamDef::new("folder", "Save into", ParamKind::Path { folder: true }))
        .param(
            ParamDef::new("file", "Project name", ParamKind::Text { multiline: false })
                .help("The project's folder and file names. Empty: the model's name."),
        )
        .param(
            ParamDef::new("magic", ".magic workspace", ParamKind::Bool)
                .help("The canvas and setup, to open and train again – never the model, which is the .szi beside it (saved with it)."),
        )
        .param(ParamDef::new("datasets", "With the datasets", ParamKind::Bool).show_if("magic", true))
        .param(ParamDef::new("pictures", "With the pictures", ParamKind::Bool).show_if("magic", true))
        .param(ParamDef::new("weights", "Weights, to train further", ParamKind::Bool).hidden())
        .param(ParamDef::new("blocks", "Blocks to use it with", ParamKind::Bool).hidden())
        .param(
            ParamDef::new("szi", "Magi Model (.szi)", ParamKind::Bool)
                .help("The model itself, with its weights and native tools: what MacroMagi and CodeMagi run."),
        )
        .param(
            ParamDef::new("onnx", "ONNX file", ParamKind::Bool)
                .help("For ONNX runtimes. Vision models run from it; Magi models do not need it."),
        )
        .param(note_param())
        .defaults(&{
            let mut v = serde_json::to_value(CompileParams::default()).expect("serializable");
            v["note"] = json!("");
            v
        })
        .summarize(|node, _| {
            let p: CompileParams = node.params_or_default();
            let place = if p.folder.trim().is_empty() {
                "No folder chosen".to_owned()
            } else {
                format!("Into {}", p.folder.rsplit(['/', '\\']).find(|s| !s.is_empty()).unwrap_or_default())
            };
            let mut formats = Vec::new();
            if p.magic {
                formats.push(".magic".to_owned());
            }
            if p.szi {
                formats.push(".szi".to_owned());
            }
            if p.onnx {
                formats.push("ONNX".to_owned());
            }
            if formats.is_empty() {
                formats.push("Nothing chosen".to_owned());
            }
            with_note(node, vec![place, formats.join(" · ")])
        }),
    );
    r.register(
        model_in(NodeDef::new(
            PROMPT,
            "Prompt Model",
            Category::Package,
            "Talk to the trained text model: write a request and see its answer, what it thought and the tools it \
             called. Wire it after Compile Package.",
        ))
        .keywords(&["prompt", "try", "test", "chat", "ask", "talk", "context", "load"])
        .param(ParamDef::new("context", "Context window", int(0, 1_048_576, "tokens")).hidden())
        .param(ParamDef::new("max_tokens", "Longest answer", int(0, 65536, "tokens")).hidden())
        .param(ParamDef::new("temperature", "Temperature", float(-1.0, 2.0, "")).hidden())
        .param(ParamDef::new("thinking", "Thinking", ParamKind::Text { multiline: false }).hidden())
        .param(ParamDef::new("effort", "Reasoning effort", ParamKind::Text { multiline: false }).hidden())
        .param(ParamDef::new("native_tools", "Run its native tools", ParamKind::Bool).hidden())
        .param(ParamDef::new("max_steps", "Tool calls per request", int(0, 200, "")).hidden())
        .param(note_param())
        .defaults(&PromptParams::default())
        .summarize(|node, _| {
            let p: PromptParams = node.params_or_default();
            let context = if p.context == 0 {
                "Context window: the model's".to_owned()
            } else {
                format!("Context window: {} tokens", p.context)
            };
            with_note(node, vec!["Ask it something".into(), context])
        }),
    );
    r.register(
        model_in(NodeDef::new(
            TEST,
            "Test Model",
            Category::Package,
            "Try the vision model: let it watch the screen live, or show it a picture, and see its answers. Wire \
             it after Compile Package.",
        ))
        .keywords(&["test", "try", "live", "watch", "picture"])
        .param(note_param())
        .defaults(&json!({ "note": "" }))
        .summarize(|node, _| with_note(node, vec!["Watch the screen".into()])),
    );
    r
}

/// A persona's tag from its name: `Gun` → `gun`.
pub fn tag_of(name: &str) -> String {
    let tag: String = name.trim().to_lowercase().chars().map(|c| if c.is_alphanumeric() { c } else { '_' }).collect();
    if tag.is_empty() { "persona".into() } else { tag }
}

// ---- setup sections -----------------------------------------------------------

/// The part of a model a setup block holds, as JSON.
pub fn section_of(kind: &str, model: &MagicModel) -> Value {
    fn to<T: Serialize>(value: &T) -> Value {
        serde_json::to_value(value).unwrap_or(Value::Null)
    }
    match kind {
        kinds::INFO => to(&InfoSection {
            name: model.name.clone(),
            description: model.description.clone(),
            author: model.author.clone(),
            license: model.license.clone(),
            source: model.source.clone(),
        }),
        kinds::TEXT => to(&model.text.clone().unwrap_or_default()),
        kinds::TOOLS | kinds::NATIVE_TOOLS => {
            let native = kind == kinds::NATIVE_TOOLS;
            let tools: Vec<&ToolSpec> =
                model.text.iter().flat_map(|t| &t.tools).filter(|t| t.native == native).collect();
            to(&tools)
        }
        kinds::THINKING => to(&model.text.as_ref().and_then(|t| t.thinking.clone()).unwrap_or_default()),
        kinds::MEMORY => to(&model.memory.clone().unwrap_or_default()),
        kinds::EFFORT => to(&model.text.as_ref().and_then(|t| t.effort.clone()).unwrap_or_default()),
        kinds::INPUT => to(&model.capture),
        kinds::ANSWERS => to(&model.classes),
        kinds::ACTION => to(&model.action),
        kinds::SETTINGS => to(&model.settings),
        kinds::PREP => to(&model.input),
        _ => Value::Null,
    }
}

fn parse<T: serde::de::DeserializeOwned + Default>(value: &Value) -> T {
    serde_json::from_value(value.clone()).unwrap_or_default()
}

/// Puts a setup block's part into a model.
pub fn apply_section(kind: &str, value: &Value, model: &mut MagicModel) {
    match kind {
        kinds::INFO => {
            let info: InfoSection = parse(value);
            (model.name, model.description, model.author, model.license, model.source) =
                (info.name, info.description, info.author, info.license, info.source);
        }
        kinds::TEXT => {
            let mut text: TextSpec = if value.is_null() { TextSpec::default() } else { parse(value) };
            // The canvas decides these; the block's copy may be stale.
            if let Some(current) = &model.text {
                text.context = current.context;
                text.personas = current.personas.clone();
            }
            text.ensure_specials();
            model.text = Some(text);
        }
        kinds::TOOLS => {
            if let Some(text) = &mut model.text
                && !value.is_null()
            {
                let tools: Vec<ToolSpec> = parse(value);
                // A Tools block from before Native Tools holds both kinds.
                if tools.iter().any(|t| t.native) {
                    text.tools = tools;
                } else {
                    text.tools.retain(|t| t.native);
                    text.tools.extend(tools);
                }
            }
        }
        kinds::NATIVE_TOOLS => {
            if let Some(text) = &mut model.text
                && !value.is_null()
            {
                let mut tools: Vec<ToolSpec> = parse(value);
                for tool in &mut tools {
                    tool.native = true;
                }
                text.tools.retain(|t| !t.native);
                tools.append(&mut text.tools);
                text.tools = tools;
            }
        }
        kinds::THINKING => {
            if let Some(text) = &mut model.text {
                text.thinking = Some(if value.is_null() { ThinkingSpec::default() } else { parse(value) });
            }
        }
        kinds::MEMORY => {
            model.memory = Some(if value.is_null() { MemorySpec::default() } else { parse(value) });
        }
        kinds::EFFORT => {
            if let Some(text) = &mut model.text {
                let mut effort: EffortSpec = if value.is_null() { EffortSpec::default() } else { parse(value) };
                effort.levels.retain(|l| !l.trim().is_empty());
                if effort.levels.is_empty() {
                    effort = EffortSpec::default();
                }
                if !effort.levels.contains(&effort.default) {
                    effort.default = effort.levels[effort.levels.len() / 2].clone();
                }
                text.effort = Some(effort);
            }
        }
        kinds::INPUT => model.capture = parse::<CaptureSpec>(value),
        kinds::ANSWERS => model.classes = parse::<Vec<ClassSpec>>(value),
        kinds::ACTION => model.action = parse::<ActionSpec>(value),
        kinds::SETTINGS => model.settings = parse::<Vec<ModelSetting>>(value),
        kinds::PREP => model.input = parse::<InputSpec>(value),
        _ => {}
    }
}

/// What a setup block's part says, in a few lines for its face.
pub fn setup_lines(kind: &str, value: &Value) -> Vec<String> {
    match kind {
        kinds::INFO => {
            let info: InfoSection = parse(value);
            let name = if info.name.trim().is_empty() { "Untitled model".to_owned() } else { info.name };
            let mut lines = vec![name];
            if !info.license.is_empty() {
                lines.push(format!("Licence: {}", info.license));
            }
            lines
        }
        kinds::TEXT => {
            let text: TextSpec = if value.is_null() { TextSpec::default() } else { parse(value) };
            let tools = match text.tools.len() {
                0 => "No tools".to_owned(),
                1 => "1 tool".to_owned(),
                n => format!("{n} tools"),
            };
            vec![tools, format!("temperature {} · up to {} tokens", text.temperature, text.max_tokens)]
        }
        kinds::TOOLS | kinds::NATIVE_TOOLS => {
            let tools: Vec<ToolSpec> = parse(value);
            let names: Vec<&str> = tools.iter().take(4).map(|t| t.name.as_str()).collect();
            let mut lines = vec![match tools.len() {
                0 => "No tools".to_owned(),
                1 => "1 tool".to_owned(),
                n => format!("{n} tools"),
            }];
            if !names.is_empty() {
                lines.push(names.join(" · ") + if tools.len() > 4 { " …" } else { "" });
            }
            lines
        }
        kinds::MEMORY => {
            let memory: MemorySpec = if value.is_null() { MemorySpec::default() } else { parse(value) };
            vec![
                format!("Hot {} MB · warm {} MB · disk {} GB", memory.hot_mb, memory.warm_mb, memory.disk_gb),
                format!(
                    "{} cache{}",
                    memory.precision.to_uppercase(),
                    if memory.persist_prefix { " · system text kept on disk" } else { "" }
                ),
            ]
        }
        kinds::THINKING => {
            let thinking: ThinkingSpec = if value.is_null() { ThinkingSpec::default() } else { parse(value) };
            vec![
                if thinking.on_by_default { "Thinks unless asked not to" } else { "Thinks only when asked" }.into(),
                "A switch where it runs".into(),
            ]
        }
        kinds::EFFORT => {
            let effort: EffortSpec = if value.is_null() { EffortSpec::default() } else { parse(value) };
            vec![effort.levels.join(" · "), format!("Default: {}", effort.default)]
        }
        kinds::INPUT => {
            let capture: CaptureSpec = parse(value);
            vec![match capture.area {
                crate::magic::CaptureArea::Center { height_share } => {
                    format!("A {:.0}-px square in the middle", height_share * 1080.0)
                }
                crate::magic::CaptureArea::Region { rect } => {
                    format!("{}×{} at {}, {}", rect.w, rect.h, rect.x, rect.y)
                }
            }]
        }
        kinds::ANSWERS => {
            let classes: Vec<ClassSpec> = parse(value);
            let mut lines: Vec<String> = classes
                .iter()
                .take(4)
                .map(|c| {
                    if c.act { format!("{} – act at {:.0}%", c.label, c.threshold * 100.0) } else { c.label.clone() }
                })
                .collect();
            if classes.len() > 4 {
                lines.push(format!("and {} more", classes.len() - 4));
            }
            if lines.is_empty() {
                lines.push("No answers yet".into());
            }
            lines
        }
        kinds::ACTION => {
            let action: ActionSpec = parse(value);
            vec![format!("Press {} for {:.0} ms", action.key.label(), action.hold_ms)]
        }
        kinds::SETTINGS => {
            let settings: Vec<ModelSetting> = parse(value);
            vec![match settings.len() {
                0 => "No sliders".into(),
                1 => "1 slider".into(),
                n => format!("{n} sliders"),
            }]
        }
        kinds::PREP => {
            let input: InputSpec = parse(value);
            vec![format!("{}×{} pictures", input.width, input.height)]
        }
        _ => Vec::new(),
    }
}

// ---- following the wires --------------------------------------------------------

fn active(graph: &Graph, node: &Node) -> bool {
    node.enabled && graph.is_active(node.id)
}

/// Brings a canvas from an earlier version up to date.
pub fn upgrade(graph: &mut Graph) {
    upgrade_native_tools(graph);
    upgrade_optimizers(graph);
}

/// Optimizer blocks from before 0.1.7.2 still holding the Muon defaults of then (0.02 for the layers,
/// no rate of their own for the embeddings) take the tuned ones; changed rates are kept.
fn upgrade_optimizers(graph: &mut Graph) {
    let untouched = |n: &Node| {
        let number = |key: &str| n.params.get(key).and_then(Value::as_f64);
        n.kind == kinds::OPTIMIZER
            && !n.params.contains_key("embed_lr")
            && n.params.get("name").and_then(Value::as_str) == Some("muon")
            && number("muon_lr") == Some(0.02)
            && number("lr") == Some(0.002)
    };
    let old: Vec<NodeId> = graph.nodes().filter(|n| untouched(n)).map(|n| n.id).collect();
    let tuned = OptimizerParams::default();
    for id in old {
        if let Some(node) = graph.node_mut(id) {
            node.set_param("muon_lr", json!(tuned.muon_lr));
            node.set_param("embed_lr", json!(tuned.embed_lr));
        }
    }
}

/// A Tools block that holds native tools (before 0.1.7) hands them to a Native Tools block,
/// chained in after it.
fn upgrade_native_tools(graph: &mut Graph) {
    if graph.nodes().any(|n| n.kind == kinds::NATIVE_TOOLS) {
        return;
    }
    let old: Vec<(NodeId, [f32; 2], Vec<ToolSpec>)> = graph
        .nodes()
        .filter(|n| n.kind == kinds::TOOLS)
        .map(|n| (n.id, n.pos, parse(&n.params_or_default::<SetupParams>().value)))
        .collect();
    let registry = registry();
    for (id, pos, tools) in old {
        let (native, regular): (Vec<ToolSpec>, Vec<ToolSpec>) = tools.into_iter().partition(|t| t.native);
        if native.is_empty() {
            continue;
        }
        let Ok(added) = graph.add(&registry, kinds::NATIVE_TOOLS, [pos[0], pos[1] + 140.0]) else { return };
        let value = |tools: &Vec<ToolSpec>| serde_json::to_value(tools).unwrap_or(Value::Null);
        if let Some(node) = graph.node_mut(added) {
            node.set_param("value", value(&native));
        }
        if let Some(node) = graph.node_mut(id) {
            node.set_param("value", value(&regular));
        }
        let next: Vec<Wire> =
            graph.wires().iter().filter(|w| w.from.node == id && w.from.port == SETUP).cloned().collect();
        for wire in next {
            graph.disconnect(&wire);
            let _ = graph.connect(&registry, PortRef::new(added, SETUP), wire.to);
        }
        let _ = graph.connect(&registry, PortRef::new(id, SETUP), PortRef::new(added, SETUP));
    }
}

/// The setup blocks chained into a block's Setup input, furthest first.
pub fn setup_chain(graph: &Graph, start: NodeId) -> Vec<NodeId> {
    let mut chain = Vec::new();
    let mut seen = BTreeSet::from([start]);
    let mut current = start;
    while let Some(source) = graph.data_source(current, SETUP) {
        if !seen.insert(source.node) {
            break;
        }
        if graph.node(source.node).is_some_and(|n| active(graph, n) && kinds::SETUP_BLOCKS.contains(&n.kind.as_str())) {
            chain.push(source.node);
        }
        current = source.node;
    }
    chain.reverse();
    chain
}

/// The start block of the canvas, when there is exactly one.
pub fn start_block(graph: &Graph) -> Result<&Node, String> {
    let starts: Vec<&Node> =
        graph.nodes().filter(|n| active(graph, n) && kinds::STARTS.contains(&n.kind.as_str())).collect();
    match starts.as_slice() {
        [] => Err("Add a block to start from: Start from Zero, Load Base Model or Use As It Is.".into()),
        [one] => Ok(*one),
        _ => Err("There is more than one start block; keep one (or turn the others off).".into()),
    }
}

/// One training stage of the pipeline.
#[derive(Clone, Debug, PartialEq)]
pub struct Stage {
    pub node: NodeId,
    pub kind: String,
    pub params: Map<String, Value>,
    /// The dataset outputs feeding it, as `(block, port)`.
    pub data: Vec<(NodeId, String)>,
    pub negatives: Vec<(NodeId, String)>,
    pub optimizer: OptimizerParams,
    /// Whether an Optimizer block is wired in (or the defaults are used).
    pub has_optimizer: bool,
}

/// The dataset files each training stage on the model wire learns from (its
/// "what to do" files), with the stage's kind, in training order.
pub fn stage_files(graph: &Graph) -> Vec<(String, Vec<String>)> {
    let Ok(found) = pipeline(graph) else { return Vec::new() };
    found
        .stages
        .iter()
        .map(|stage| {
            let mut files: Vec<String> = Vec::new();
            for (node, port) in &stage.data {
                let Some(block) = graph.node(*node) else { continue };
                let params: TextDataParams = block.params_or_default();
                let list = if port == "negative" { params.negative } else { params.positive };
                for file in list {
                    if !files.contains(&file) {
                        files.push(file);
                    }
                }
            }
            (stage.kind.clone(), files)
        })
        .collect()
}

/// The model wire, followed from its start to its end.
#[derive(Clone, Debug, PartialEq)]
pub struct Pipeline {
    pub entry: Node,
    pub architecture: Architecture,
    pub personas: Option<PersonaParams>,
    pub stages: Vec<Stage>,
    pub compile: Option<(NodeId, CompileParams)>,
    /// Every block on the model wire, in order (start first).
    pub path: Vec<NodeId>,
}

impl Pipeline {
    pub fn kind(&self, model: &MagicModel) -> ModelKind {
        match self.entry.kind.as_str() {
            kinds::ZERO if self.entry.params_or_default::<ZeroParams>().kind == "vision" => ModelKind::Vision,
            kinds::ZERO => ModelKind::Text,
            _ if self.architecture.conv.is_some() => ModelKind::Vision,
            _ if self.stages.iter().any(|s| s.kind == kinds::VISION_TRAIN) => ModelKind::Vision,
            _ if self.stages.iter().any(|s| s.kind != kinds::VISION_TRAIN) => ModelKind::Text,
            _ => model.kind,
        }
    }

    pub fn trains(&self) -> bool {
        !self.stages.is_empty()
    }
}

/// The blocks after a block on the model wire.
fn next_on_wire(graph: &Graph, node: NodeId) -> Vec<&Node> {
    graph
        .wires()
        .iter()
        .filter(|w| w.from.node == node && w.from.port == MODEL && w.to.port == MODEL)
        .filter_map(|w| graph.node(w.to.node))
        .filter(|n| active(graph, n))
        .collect()
}

/// The feed-forward layer of a layer block: the block wired in, or a plain one.
fn ffn_spec(graph: &Graph, node: &Node, mult: f64) -> (Value, Option<ArchBlock>, Option<PersonaParams>) {
    let provider = graph.data_source(node.id, FFN).and_then(|p| graph.node(p.node)).filter(|n| active(graph, n));
    match provider {
        Some(p) if p.kind == kinds::MOE => {
            let m: MoeParams = p.params_or_default();
            let spec = json!({ "kind": "moe", "experts": m.experts, "shared": m.shared, "top_k": m.top_k,
                               "mult": m.mult, "gating": m.gating, "balance": m.balance, "hash": m.hash });
            (spec, Some(ArchBlock { kind: p.kind.clone(), params: p.params.clone(), ffn: None }), None)
        }
        Some(p) if p.kind == kinds::PERSONA => {
            let persona: PersonaParams = p.params_or_default();
            let spec = json!({ "kind": "persona", "mult": persona.mult });
            (spec, Some(ArchBlock { kind: p.kind.clone(), params: p.params.clone(), ffn: None }), Some(persona))
        }
        _ => (json!({ "kind": "dense", "mult": mult }), None, None),
    }
}

/// Follows the model wire from the start through the architecture blocks,
/// the training stages and the package.
pub fn pipeline(graph: &Graph) -> Result<Pipeline, String> {
    use kinds::*;
    let entry = start_block(graph)?;
    let mut arch = Architecture::default();
    let mut size_seen = false;
    let mut pipeline = Pipeline {
        entry: entry.clone(),
        architecture: Architecture::default(),
        personas: None,
        stages: Vec::new(),
        compile: None,
        path: vec![entry.id],
    };
    let mut current = entry.id;
    let mut seen = BTreeSet::from([current]);
    let mut stage_seen = false;
    loop {
        let next = next_on_wire(graph, current);
        let onward: Vec<&Node> = next
            .iter()
            .copied()
            .filter(|n| {
                ARCHITECTURE.contains(&n.kind.as_str()) || STAGES.contains(&n.kind.as_str()) || n.kind == COMPILE
            })
            .collect();
        let node = match onward.as_slice() {
            [] => break,
            [one] => *one,
            _ => return Err("The Model wire splits into two paths; a model is built along one line.".into()),
        };
        if !seen.insert(node.id) {
            break;
        }
        current = node.id;
        pipeline.path.push(node.id);
        let kind = node.kind.as_str();
        if ARCHITECTURE.contains(&kind) && stage_seen {
            return Err("Put the architecture blocks before the training stages.".into());
        }
        if ARCHITECTURE.contains(&kind) && pipeline.compile.is_some()
            || STAGES.contains(&kind) && pipeline.compile.is_some()
        {
            return Err("Compile Package comes last on the Model wire.".into());
        }
        let record = |ffn: Option<ArchBlock>| ArchBlock {
            kind: node.kind.clone(),
            params: node.params.clone(),
            ffn: ffn.map(Box::new),
        };
        match kind {
            SIZE => {
                let p: SizeParams = node.params_or_default();
                (arch.width, arch.heads, arch.kv_heads, arch.context) = (p.width, p.heads, p.kv_heads, p.context);
                arch.positions = p.positions.clone();
                arch.rope_theta = p.rope_theta;
                (arch.tied, arch.embed_scale) = (p.tied, p.embed_scale);
                size_seen = true;
                arch.blocks.push(record(None));
            }
            CONTEXT => {
                let p: ContextParams = node.params_or_default();
                if p.train > 0 {
                    arch.context = p.train;
                }
                arch.rope_scaling = Some(p.scaling(arch.context));
                arch.blocks.push(record(None));
            }
            HYPER => {
                let p: HyperParams = node.params_or_default();
                arch.hyper_connections = Some(json!({ "streams": p.streams, "sinkhorn": p.sinkhorn }));
                arch.blocks.push(record(None));
            }
            MTP => {
                arch.mtp = Some(json!({ "weight": node.params_or_default::<MtpParams>().weight }));
                arch.blocks.push(record(None));
            }
            SPLIT => {
                arch.prefix_split = true;
                arch.blocks.push(record(None));
            }
            CONV => {
                let p: ConvParams = node.params_or_default();
                arch.conv = Some(json!({ "layers": p.layers, "channels": p.channels }));
                arch.blocks.push(record(None));
            }
            LAYERS | WINDOW | MLA | CSA | HCA | SSM => {
                let layers = node.param::<u32>("layers").unwrap_or(1).max(1);
                let mult = node.param::<f64>("mult").unwrap_or(2.75);
                let (ffn, provider, persona) = ffn_spec(graph, node, mult);
                if persona.is_some() {
                    pipeline.personas = persona;
                }
                let options: LayersParams = node.params_or_default();
                let attention = match kind {
                    LAYERS => json!({ "kind": "full" }),
                    WINDOW => json!({ "kind": "window", "window": node.param::<u32>("window").unwrap_or(64) }),
                    MLA => {
                        let p: MlaParams = node.params_or_default();
                        json!({ "kind": "mla", "kv_rank": p.kv_rank, "q_rank": p.q_rank, "rope_dim": p.rope_dim })
                    }
                    CSA => {
                        let p: CsaParams = node.params_or_default();
                        json!({ "kind": "csa", "compress": p.compress, "top_k": p.top_k, "window": p.window,
                                "indexer_heads": p.indexer_heads, "indexer_dim": p.indexer_dim, "rope_dim": p.rope_dim })
                    }
                    HCA => {
                        let p: HcaParams = node.params_or_default();
                        json!({ "kind": "hca", "compress": p.compress, "window": p.window, "rope_dim": p.rope_dim })
                    }
                    _ => json!({ "kind": "ssm", "state": node.param::<u32>("state").unwrap_or(16) }),
                };
                for _ in 0..layers {
                    if kind == LAYERS {
                        let mut ffn = ffn.clone();
                        if options.activation == "geglu" && ffn["kind"] == "dense" {
                            ffn["activation"] = json!("geglu");
                        }
                        let mut layer = json!({ "attention": options.attention(arch.layers.len()), "ffn": ffn });
                        if options.norm == "sandwich" {
                            layer["norm"] = json!("sandwich");
                        }
                        arch.layers.push(layer);
                    } else {
                        arch.layers.push(json!({ "attention": attention, "ffn": ffn }));
                    }
                }
                arch.blocks.push(record(provider));
            }
            COMPILE => pipeline.compile = Some((node.id, node.params_or_default())),
            _ => {
                stage_seen = true;
                let source = |port: &str| {
                    graph.data_source(node.id, port).map(|p| vec![(p.node, p.port.clone())]).unwrap_or_default()
                };
                let optimizer =
                    graph.data_source(node.id, OPTIMIZER).and_then(|p| graph.node(p.node)).filter(|n| active(graph, n));
                pipeline.stages.push(Stage {
                    node: node.id,
                    kind: node.kind.clone(),
                    params: node.params.clone(),
                    data: source(if kind == VISION_TRAIN { "pictures" } else { "data" }),
                    negatives: source("negative"),
                    optimizer: optimizer.map(|n| n.params_or_default()).unwrap_or_default(),
                    has_optimizer: optimizer.is_some(),
                });
            }
        }
    }
    if !size_seen && pipeline.trains() && entry.kind == ZERO {
        // Without a Model Size block, the defaults.
        let p = SizeParams::default();
        (arch.width, arch.heads, arch.kv_heads, arch.context) = (p.width, p.heads, p.kv_heads, p.context);
    }
    if arch.heads == 0 || arch.width % arch.heads != 0 {
        return Err(format!(
            "Model Size: the width ({}) must divide by the attention heads ({}).",
            arch.width, arch.heads
        ));
    }
    if arch.kv_heads == 0 || arch.heads % arch.kv_heads != 0 {
        return Err(format!(
            "Model Size: the heads ({}) must divide by the key/value heads ({}).",
            arch.heads, arch.kv_heads
        ));
    }
    if entry.kind == CONFIG && pipeline.trains() {
        return Err("Use As It Is does not train. To train, start from zero or from a base model's weights.".into());
    }
    pipeline.architecture = arch;
    Ok(pipeline)
}

// ---- the model from the canvas -------------------------------------------------

/// Builds the model's settings from the canvas – the setup blocks chained into
/// the start, the architecture's context and personas – and writes notes on
/// blocks that are not connected where they would be used.
pub fn derive(graph: &mut Graph, model: &mut MagicModel, has_onnx: bool) {
    let start = start_block(graph).map(|n| (n.id, n.kind.clone(), n.params.clone())).ok();
    let pipeline = pipeline(graph).ok();
    // The kind: a model not trained yet is what its Start from Zero says.
    if let Some((_, kind, params)) = &start
        && kind == kinds::ZERO
        && !has_onnx
    {
        let zero: ZeroParams = serde_json::from_value(Value::Object(params.clone())).unwrap_or_default();
        model.kind = if zero.kind == "vision" { ModelKind::Vision } else { ModelKind::Text };
    }
    // The context and personas come from the architecture.
    if model.kind == ModelKind::Text {
        let mut text = model.text.clone().unwrap_or_default();
        if let Some(p) = &pipeline
            && p.trains()
        {
            text.context = p.architecture.context;
        }
        text.personas = pipeline.as_ref().and_then(|p| p.personas.clone()).map_or_else(Vec::new, |p| {
            vec![
                Persona { name: p.first.clone(), tag: format!("<{}>", tag_of(&p.first)), role: p.first_role },
                Persona { name: p.second.clone(), tag: format!("<{}>", tag_of(&p.second)), role: p.second_role },
            ]
        });
        text.ensure_specials();
        model.text = Some(text);
    }
    // The setup chained into the start; what is not chained in is the default.
    let chain = start.as_ref().map(|(id, ..)| setup_chain(graph, *id)).unwrap_or_default();
    for kind in kinds::SETUP_BLOCKS {
        let used = chain.iter().rev().find_map(|id| graph.node(*id).filter(|n| n.kind == kind));
        // Without a Tools block the tools are the Text block's own (as files from before it had them).
        if kind == kinds::TOOLS && used.is_none() {
            continue;
        }
        // A Native Tools block taken out of the chain takes its tools with it.
        if kind == kinds::NATIVE_TOOLS && used.is_none() {
            if graph.nodes().any(|n| n.kind == kinds::NATIVE_TOOLS)
                && let Some(text) = &mut model.text
            {
                text.tools.retain(|t| !t.native);
            }
            continue;
        }
        if kinds::SWITCHES.contains(&kind) && used.is_none() {
            if kind == kinds::MEMORY {
                model.memory = None;
            } else if let Some(text) = &mut model.text {
                match kind {
                    kinds::THINKING => text.thinking = None,
                    _ => text.effort = None,
                }
            }
            continue;
        }
        let value = used.map(|n| n.params_or_default::<SetupParams>().value).unwrap_or(Value::Null);
        apply_section(kind, &value, model);
    }
    // A model that is not text has no text settings.
    if model.kind == ModelKind::Vision {
        model.text = None;
    }
    write_notes(graph, model, start.as_ref().map(|s| s.0), pipeline.as_ref(), &chain);
}

fn write_notes(
    graph: &mut Graph,
    model: &MagicModel,
    start: Option<NodeId>,
    pipeline: Option<&Pipeline>,
    chain: &[NodeId],
) {
    use kinds::*;
    let text = model.kind == ModelKind::Text;
    let on_path: BTreeSet<NodeId> = pipeline.map(|p| p.path.iter().copied().collect()).unwrap_or_default();
    let mut notes: Vec<(NodeId, String)> = Vec::new();
    let has_output =
        |graph: &Graph, id: NodeId, port: &str| graph.wires().iter().any(|w| w.from.node == id && w.from.port == port);
    for node in graph.nodes() {
        let kind = node.kind.as_str();
        let note = if SETUP_BLOCKS.contains(&kind) {
            let vision_only = [INPUT, ANSWERS, ACTION, SETTINGS, PREP].contains(&kind);
            if !chain.contains(&node.id) {
                "Not used: chain it into the start's Setup input".into()
            } else if text && vision_only {
                "Only vision models use this".into()
            } else if !text && kind == TEXT {
                "Only text models use this".into()
            } else {
                String::new()
            }
        } else if STARTS.contains(&kind) {
            if Some(node.id) != start && start.is_some() {
                String::new()
            } else if chain.is_empty() && graph.data_source(node.id, SETUP).is_none() {
                "Plug Model Info into Setup to name it".into()
            } else {
                String::new()
            }
        } else if kinds::LAYER_BLOCKS.contains(&kind) || STAGES.contains(&kind) || kind == COMPILE {
            if !on_path.contains(&node.id) {
                "Not on the Model wire".into()
            } else if STAGES.contains(&kind) {
                let data_port = if kind == VISION_TRAIN { "pictures" } else { "data" };
                if graph.data_source(node.id, data_port).is_none() {
                    if kind == VISION_TRAIN {
                        "Wire a Vision Dataset into Pictures".into()
                    } else {
                        "Wire a Text Dataset into it".into()
                    }
                } else if graph.data_source(node.id, OPTIMIZER).is_none() {
                    "Default optimizer (AdamW)".into()
                } else {
                    String::new()
                }
            } else if kind == COMPILE && node.param::<String>("folder").is_none_or(|f| f.trim().is_empty()) {
                "Choose a folder to save into".into()
            } else {
                String::new()
            }
        } else if [MOE, PERSONA].contains(&kind) {
            if has_output(graph, node.id, FFN) {
                String::new()
            } else {
                "Wire it into a layer block's Feed-forward".into()
            }
        } else if kind == OPTIMIZER {
            if has_output(graph, node.id, OPTIMIZER) { String::new() } else { "Wire it into a training stage".into() }
        } else if kind == TEXT_DATA {
            let p: TextDataParams = node.params_or_default();
            if p.positive.is_empty() && p.negative.is_empty() {
                "No files yet: select it to add some".into()
            } else if !has_output(graph, node.id, "positive") && !has_output(graph, node.id, "negative") {
                "Wire it into a training stage".into()
            } else {
                String::new()
            }
        } else if kind == IMAGE_DATA {
            if model.training.is_empty() && model.finetune.is_empty() {
                "No pictures yet: select it to add some".into()
            } else if !has_output(graph, node.id, "pictures") {
                "Wire it into Vision Training".into()
            } else {
                String::new()
            }
        } else if [PROMPT, TEST].contains(&kind) {
            if (kind == PROMPT) != text {
                if text { "Use Prompt Model for text models".into() } else { "Use Test Model for vision models".into() }
            } else {
                String::new()
            }
        } else {
            continue;
        };
        notes.push((node.id, note));
    }
    for (id, note) in notes {
        if let Some(node) = graph.node_mut(id)
            && node.param::<String>("note").unwrap_or_default() != note
        {
            node.set_param("note", note);
        }
    }
}

// ---- the canvas a model opens with --------------------------------------------

struct Builder {
    registry: Registry,
    nodes: Vec<Node>,
    wires: Vec<Wire>,
}

impl Builder {
    fn new() -> Self {
        Self { registry: registry(), nodes: Vec::new(), wires: Vec::new() }
    }

    fn add(&mut self, kind: &str, pos: [f32; 2], params: Value) -> NodeId {
        let id = self.nodes.len() as NodeId + 1;
        let mut node =
            Node::new(id, kind, pos, self.registry.get(kind).map(|d| d.default_params()).unwrap_or_default());
        if let Value::Object(map) = params {
            node.params.extend(map);
        }
        self.nodes.push(node);
        id
    }

    fn wire(&mut self, from: NodeId, out: &str, to: NodeId, input: &str) {
        self.wires.push(Wire { from: PortRef::new(from, out), to: PortRef::new(to, input) });
    }
}

const STEP: f32 = 250.0;

/// The canvas a model opens with when it has none: its setup chained into a
/// start, and a pipeline that fits it – training from zero for a new model,
/// training further for one with weights, using it as it is for one made
/// elsewhere.
pub fn default_canvas(model: &MagicModel, has_weights: bool, has_onnx: bool, datasets: &[(String, bool)]) -> Fragment {
    default_canvas_for(model, has_weights, has_onnx, datasets, &MAGI)
}

/// The canvas a model starts with; a new text model is Magi at `size`.
pub fn default_canvas_for(
    model: &MagicModel,
    has_weights: bool,
    has_onnx: bool,
    datasets: &[(String, bool)],
    size: &MagiSize,
) -> Fragment {
    use kinds::*;
    let mut b = Builder::new();
    let text = model.kind == ModelKind::Text;
    let row = 0.0;

    // The setup, chained into the start (below the pipeline).
    let mut setup_kinds: Vec<&str> =
        if text { vec![INFO, TEXT, TOOLS] } else { vec![INFO, ANSWERS, INPUT, ACTION, SETTINGS, PREP] };
    if model.text.as_ref().is_some_and(|t| text && t.tools.iter().any(|tool| tool.native)) {
        setup_kinds.push(NATIVE_TOOLS);
    }
    if let Some(spec) = model.text.as_ref().filter(|_| text) {
        if spec.thinking.is_some() {
            setup_kinds.push(THINKING);
        }
        if spec.effort.is_some() {
            setup_kinds.push(EFFORT);
        }
    }
    if model.memory.is_some() {
        setup_kinds.push(MEMORY);
    }
    let mut setup = BTreeSet::new();
    let mut previous: Option<NodeId> = None;
    for (index, kind) in setup_kinds.iter().enumerate() {
        let (column, line) = (index % 3, index / 3);
        let id = b.add(
            kind,
            [column as f32 * STEP - 3.0 * STEP, 260.0 + line as f32 * 140.0],
            json!({ "value": section_of(kind, model) }),
        );
        if let Some(prev) = previous {
            b.wire(prev, SETUP, id, SETUP);
        }
        previous = Some(id);
        setup.insert(id);
    }

    let entry = if has_weights {
        b.add(BASE, [0.0, row], json!({}))
    } else if has_onnx {
        b.add(CONFIG, [0.0, row], json!({}))
    } else {
        b.add(ZERO, [0.0, row], json!({ "kind": if text { "text" } else { "vision" } }))
    };
    if let Some(last) = previous {
        b.wire(last, SETUP, entry, SETUP);
    }
    let mut last = entry;
    let mut x = STEP;
    let place = |b: &mut Builder, kind: &str, params: Value, last: &mut NodeId, x: &mut f32| -> NodeId {
        let id = b.add(kind, [*x, row], params);
        b.wire(*last, MODEL, id, MODEL);
        *last = id;
        *x += STEP;
        id
    };

    if has_weights || !has_onnx {
        // The architecture: the model's own, Magi for text, a small one for vision.
        let blocks: Vec<ArchBlock> = match &model.architecture {
            Some(a) if !a.blocks.is_empty() => a.blocks.clone(),
            _ if text => magi_blocks_of(size),
            _ => vec![
                ArchBlock {
                    kind: SIZE.into(),
                    params: obj(
                        json!({ "width": 64, "heads": 4, "kv_heads": 4, "context": model.input.width.clamp(16, 512) }),
                    ),
                    ffn: None,
                },
                ArchBlock { kind: CONV.into(), params: Map::new(), ffn: None },
            ],
        };
        let mut providers: Vec<(NodeId, ArchBlock)> = Vec::new();
        for block in &blocks {
            let id = place(&mut b, &block.kind, Value::Object(block.params.clone()), &mut last, &mut x);
            if let Some(ffn) = &block.ffn {
                let existing =
                    providers.iter().find(|(_, p)| p.kind == ffn.kind && p.params == ffn.params).map(|(id, _)| *id);
                let provider = existing.unwrap_or_else(|| {
                    let provider_id = b.add(&ffn.kind, [x - STEP, row - 190.0], Value::Object(ffn.params.clone()));
                    providers.push((provider_id, (**ffn).clone()));
                    provider_id
                });
                b.wire(provider, FFN, id, FFN);
            }
        }
        if text {
            // Magi's recipe. Pretraining (from zero): language and code from the web, read where it
            // trains. Then fine-tuning on agent work and conversations, rewriting long contexts short
            // (compaction), and reinforcement for good tool use.
            let jsonl: Vec<String> =
                datasets.iter().filter(|(f, n)| !n && is_jsonl(f)).map(|(f, _)| f.clone()).collect();
            let negative: Vec<String> = datasets.iter().filter(|(_, n)| *n).map(|(f, _)| f.clone()).collect();
            if !has_weights {
                let language: Vec<String> =
                    datasets.iter().filter(|(f, n)| !n && !is_jsonl(f)).map(|(f, _)| f.clone()).collect();
                let files = b.add(TEXT_DATA, [x, row + 670.0], json!({ "positive": language, "negative": [] }));
                b.nodes.last_mut().expect("added").label = "Language (example files)".into();
                let code = b.add(
                    CORPUS,
                    [x, row + 450.0],
                    json!({ "repo": "OpenCoder-LLM/opc-fineweb-code-corpus", "files": "data/train-*.parquet",
                            "field": "text", "tokens": size.code_tokens }),
                );
                b.nodes.last_mut().expect("added").label = "Code on the web (pretraining)".into();
                b.wire(files, "positive", code, "with");
                let web = b.add(CORPUS, [x, row + 230.0], json!({ "tokens": size.web_tokens }));
                b.nodes.last_mut().expect("added").label = "FineWeb-Edu (pretraining)".into();
                b.wire(code, "positive", web, "with");
                // Muon, as DeepSeek-V4 trains: 16 rows of 8,192 tokens a step, four steps a weight update.
                let optimizer = b.add(
                    OPTIMIZER,
                    [x + STEP * 0.5, row - 210.0],
                    json!({ "name": "muon", "muon_lr": 0.002, "embed_lr": 0.006, "warmup": size.warmup, "accumulate": 4 }),
                );
                b.nodes.last_mut().expect("added").label = "Optimizer (pretraining)".into();
                // The corpora once over: three billion tokens for Magi.
                let pretrain =
                    place(&mut b, PRETRAIN, json!({ "steps": size.pretrain_steps, "batch": 16 }), &mut last, &mut x);
                b.wire(web, "positive", pretrain, "data");
                b.wire(optimizer, OPTIMIZER, pretrain, OPTIMIZER);
            }
            let work = b.add(TEXT_DATA, [x, row + 450.0], json!({ "positive": jsonl, "negative": negative }));
            b.nodes.last_mut().expect("added").label = "Agent work (fine-tuning)".into();
            let chats = b.add(
                CORPUS,
                [x, row + 230.0],
                json!({ "repo": "HuggingFaceTB/smol-smoltalk", "files": "data/train-*.parquet", "field": "messages",
                        "tokens": 200_000_000u64 }),
            );
            b.nodes.last_mut().expect("added").label = "Conversations (fine-tuning)".into();
            b.wire(work, "positive", chats, "with");
            let tuning = b.add(
                OPTIMIZER,
                [x + STEP, row - 210.0],
                json!({ "name": "muon", "muon_lr": 0.001, "embed_lr": 0.003, "warmup": 100 }),
            );
            b.nodes.last_mut().expect("added").label = "Optimizer (fine-tuning)".into();
            let finetune =
                place(&mut b, FINETUNE, json!({ "steps": size.finetune_steps, "batch": 8 }), &mut last, &mut x);
            b.wire(chats, "positive", finetune, "data");
            b.wire(work, "negative", finetune, "negative");
            b.wire(tuning, OPTIMIZER, finetune, OPTIMIZER);
            let compaction =
                place(&mut b, COMPACTION, json!({ "steps": size.compaction_steps, "batch": 8 }), &mut last, &mut x);
            b.wire(work, "positive", compaction, "data");
            b.wire(tuning, OPTIMIZER, compaction, OPTIMIZER);
            let rl = place(
                &mut b,
                RL,
                json!({ "steps": size.rl_steps, "group": 8, "max_new": 384, "temperature": 0.8, "rules": agent_rules() }),
                &mut last,
                &mut x,
            );
            b.wire(work, "positive", rl, "data");
            let gentle = b.add(
                OPTIMIZER,
                [x - STEP * 0.5, row - 210.0],
                json!({ "name": "adamw", "lr": 0.00002, "warmup": 10 }),
            );
            b.nodes.last_mut().expect("added").label = "Optimizer (gentle)".into();
            b.wire(gentle, OPTIMIZER, rl, OPTIMIZER);
        } else {
            let optimizer = b.add(OPTIMIZER, [x + STEP * 0.5, row - 210.0], json!({ "name": "muon" }));
            let data = b.add(IMAGE_DATA, [x, row + 230.0], json!({}));
            let id = place(&mut b, VISION_TRAIN, json!({}), &mut last, &mut x);
            b.wire(data, "pictures", id, "pictures");
            b.wire(optimizer, OPTIMIZER, id, OPTIMIZER);
        }
    }
    let compile = place(&mut b, COMPILE, json!({}), &mut last, &mut x);
    let try_kind = if text { PROMPT } else { TEST };
    let try_it = b.add(try_kind, [x, row], json!({}));
    b.wire(compile, MODEL, try_it, MODEL);

    let group =
        Group { id: 1, name: "The model's setup".into(), color: None, nodes: setup, parent: None, enabled: true };
    Fragment { nodes: b.nodes, wires: b.wires, groups: vec![group] }
}

fn obj(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        _ => Map::new(),
    }
}

// ---- common architectures -------------------------------------------------------

/// An architecture to start from: blocks that replace the canvas's own, and
/// setup blocks it adds to the setup chain (Thinking, Reasoning Effort).
pub struct Common {
    pub name: &'static str,
    pub description: &'static str,
    pub blocks: Vec<ArchBlock>,
    /// Setup blocks it adds, with their part of the model (`Null`: the block's default).
    pub setup: Vec<(&'static str, Value)>,
}

/// Magi's thinking: on – it plans before it acts, and thinks again between tool calls.
pub fn magi_thinking() -> Value {
    json!({ "on_by_default": true })
}

/// Magi's reasoning effort: standard, or superior.
pub fn magi_effort() -> EffortSpec {
    EffortSpec { levels: vec!["standard".into(), "superior".into()], default: "standard".into() }
}

/// Magi: MacroMagi's one text architecture – an agent that knows how to work, not facts, and
/// looks up what it needs with its tools. About 418 million parameters, 148 million of them at
/// work for each token:
///
/// - 768 wide and 32 layers deep (deep and narrow is best at this size), twelve heads of 64
///   sharing four key/value heads, queries and keys normalized, a tied output head;
/// - three of every four layers look only at their block of 1,024 tokens and the block before, the fourth
///   at everything:
///   long tool transcripts stay cheap to read and to keep in memory (Gemma 3, gpt-oss);
/// - after three plain layers, every feed-forward part is DeepSeekMoE: a shared expert and
///   sixteen small routed ones, two chosen per token by √Softplus affinities and balanced by a
///   bias (DeepSeek-V3/V4) – the knowledge of a 418M model at the cost of a 148M one;
/// - DeepSeek-V3's multi-token prediction while training (it predicts the token after next too);
/// - 8,192 tokens while training, stretched by YaRN to 128K when it runs.
pub fn magi_blocks() -> Vec<ArchBlock> {
    magi_blocks_of(&MAGI)
}

/// A size of Magi: the same design – windows, experts, multi-token prediction – at a width and
/// depth, and the recipe scaled to it.
pub struct MagiSize {
    pub name: &'static str,
    pub width: u32,
    pub heads: u32,
    pub kv_heads: u32,
    /// All the layers: three plain ones, then layers with experts.
    pub layers: u32,
    pub pretrain_steps: u64,
    /// Tokens of FineWeb-Edu and of code to pretrain on (16 rows of 8,192 tokens a step reads them once).
    pub web_tokens: u64,
    pub code_tokens: u64,
    pub warmup: u64,
    pub finetune_steps: u64,
    pub compaction_steps: u64,
    pub rl_steps: u64,
}

/// Magi: about 418M parameters, 148M at work per token; three billion tokens of pretraining.
pub const MAGI: MagiSize = MagiSize {
    name: "Magi",
    width: 768,
    heads: 12,
    kv_heads: 4,
    layers: 32,
    pretrain_steps: 22_900,
    web_tokens: 2_250_000_000,
    code_tokens: 750_000_000,
    warmup: 1000,
    finetune_steps: 4_000,
    compaction_steps: 400,
    rl_steps: 300,
};

/// Magi at about 150M parameters (64M at work per token), on 750 million tokens: the whole recipe
/// in a few hours on a GPU server, to try it before the full size.
pub const MAGI_TEST: MagiSize = MagiSize {
    name: "Magi 150M",
    width: 512,
    heads: 8,
    kv_heads: 4,
    layers: 24,
    pretrain_steps: 5_700,
    web_tokens: 560_000_000,
    code_tokens: 190_000_000,
    warmup: 300,
    finetune_steps: 1_000,
    compaction_steps: 100,
    rl_steps: 100,
};

/// Magi's architecture blocks at `size`.
pub fn magi_blocks_of(size: &MagiSize) -> Vec<ArchBlock> {
    use kinds::*;
    let layers = |count: u32| json!({ "layers": count, "mult": 2.75, "more": true, "qk_norm": true, "window": 1024, "global_every": 4 });
    let experts = block(
        MOE,
        json!({ "experts": 16, "shared": 1, "top_k": 2, "mult": 0.375, "gating": "sqrt_softplus", "balance": "bias" }),
    );
    vec![
        block(
            SIZE,
            json!({ "width": size.width, "heads": size.heads, "kv_heads": size.kv_heads, "context": 8192,
                    "rope_theta": 500000.0, "tied": true }),
        ),
        block(MTP, json!({ "weight": 0.3 })),
        block(LAYERS, layers(3)),
        with_ffn(block(LAYERS, layers(size.layers - 3)), experts),
        block(CONTEXT, json!({ "train": "0", "target": "131072", "method": "yarn" })),
    ]
}

fn block(kind: &str, params: Value) -> ArchBlock {
    ArchBlock { kind: kind.into(), params: obj(params), ffn: None }
}

fn with_ffn(mut b: ArchBlock, ffn: ArchBlock) -> ArchBlock {
    b.ffn = Some(Box::new(ffn));
    b
}

/// The architectures Presets › Commons offers: Magi, and Magi 150M to try a recipe quickly.
pub fn commons() -> Vec<Common> {
    let setup = || {
        vec![(kinds::THINKING, magi_thinking()), (kinds::EFFORT, json!(magi_effort())), (kinds::MEMORY, Value::Null)]
    };
    vec![
        Common {
            name: "Magi",
            description: "MacroMagi's text architecture: about 418M parameters, 148M at work per token. 32 layers, \
                      three in four looking at the last 1,024–2,048 tokens, DeepSeekMoE experts, multi-token prediction \
                      while training, 8K tokens in training and 128K when it runs. It trains on a GPU server \
                      (File › Export as Python) and runs in MacroMagi.",
            blocks: magi_blocks(),
            setup: setup(),
        },
        Common {
            name: "Magi 150M",
            description: "Magi's design at about 150M parameters, 64M at work per token: 512 wide, 24 layers. \
                          To try a recipe in a few hours before training the full size.",
            blocks: magi_blocks_of(&MAGI_TEST),
            setup: setup(),
        },
    ]
}

/// What reinforcement learning rewards in an agent: finishing, calling tools well – the tool the
/// dataset's answer calls first, one that is offered, in a well-formed call – and never writing a
/// tool's result itself or going round in circles.
pub fn agent_rules() -> Value {
    json!([
        { "kind": "ends", "score": 1.0 },
        { "kind": "same_tool", "score": 2.0 },
        { "kind": "tool_valid", "score": 1.0 },
        { "kind": "tool_malformed", "score": -2.0 },
        { "kind": "unknown_tool", "score": -2.0 },
        { "kind": "made_up_result", "score": -3.0 },
        { "kind": "loop", "score": -3.0, "tokens": 4 },
    ])
}

/// Puts a common architecture on the canvas in place of its architecture
/// blocks (or, without a pipeline, at `at`). Returns the new blocks.
pub fn apply_common(graph: &mut Graph, common: &Common, at: [f32; 2]) -> Vec<NodeId> {
    use kinds::*;
    let reg = registry();
    let defaults = |kind: &str| reg.get(kind).map(|d| d.default_params()).unwrap_or_default();
    // Where the architecture goes: after the start, before what followed it.
    let path = pipeline(graph).map(|p| p.path).unwrap_or_default();
    let start = path.first().copied();
    let old: Vec<NodeId> = path
        .iter()
        .copied()
        .filter(|id| graph.node(*id).is_some_and(|n| ARCHITECTURE.contains(&n.kind.as_str())))
        .collect();
    let after = path.iter().copied().find(|id| {
        graph.node(*id).is_some_and(|n| !ARCHITECTURE.contains(&n.kind.as_str()) && !STARTS.contains(&n.kind.as_str()))
    });
    let origin = start.and_then(|id| graph.node(id)).map(|n| [n.pos[0] + STEP, n.pos[1]]).unwrap_or(at);
    // Feed-forward blocks only the old architecture used go too.
    let providers: Vec<NodeId> = old
        .iter()
        .filter_map(|id| graph.data_source(*id, FFN).map(|p| p.node))
        .filter(|p| graph.wires().iter().filter(|w| w.from.node == *p).all(|w| old.contains(&w.to.node)))
        .collect();
    let old_width = old.len() as f32 * STEP;
    for id in old.iter().chain(&providers) {
        graph.remove(*id);
    }
    let new_width = common.blocks.len() as f32 * STEP;
    // Blocks to the right move over to make room.
    let shift = new_width - old_width;
    if start.is_some() {
        for node in graph.nodes_mut() {
            if node.pos[0] >= origin[0] - 1.0 {
                node.pos[0] += shift;
            }
        }
    }
    let mut created = Vec::new();
    let mut previous = start;
    let mut made: Vec<(ArchBlock, NodeId)> = Vec::new();
    for (index, b) in common.blocks.iter().enumerate() {
        let mut params = defaults(&b.kind);
        params.extend(b.params.clone());
        let id = graph.add_with(&b.kind, [origin[0] + index as f32 * STEP, origin[1]], params);
        created.push(id);
        if let Some(prev) = previous {
            let _ = graph.connect(&reg, PortRef::new(prev, MODEL), PortRef::new(id, MODEL));
        }
        previous = Some(id);
        if let Some(ffn) = &b.ffn {
            let provider = made.iter().find(|(p, _)| p.kind == ffn.kind && p.params == ffn.params).map(|(_, id)| *id);
            let provider = provider.unwrap_or_else(|| {
                let mut params = defaults(&ffn.kind);
                params.extend(ffn.params.clone());
                let pid = graph.add_with(&ffn.kind, [origin[0] + index as f32 * STEP, origin[1] - 190.0], params);
                made.push(((**ffn).clone(), pid));
                created.push(pid);
                pid
            });
            let _ = graph.connect(&reg, PortRef::new(provider, FFN), PortRef::new(id, FFN));
        }
    }
    if let (Some(last), Some(next)) = (previous, after) {
        let _ = graph.connect(&reg, PortRef::new(last, MODEL), PortRef::new(next, MODEL));
    }
    // Setup blocks it needs join the end of the setup chain, unless they are in it.
    if let Some(start) = start {
        for (kind, value) in &common.setup {
            let chain = setup_chain(graph, start);
            if chain.iter().any(|id| graph.node(*id).is_some_and(|n| n.kind == *kind)) {
                continue;
            }
            let last = graph.data_source(start, SETUP).map(|p| p.node);
            let below = last
                .and_then(|id| graph.node(id))
                .map_or([origin[0] - STEP, origin[1] + 260.0], |n| [n.pos[0] + STEP, n.pos[1]]);
            let mut params = defaults(kind);
            if !value.is_null() {
                params.insert("value".into(), value.clone());
            }
            let id = graph.add_with(kind, below, params);
            created.push(id);
            if let Some(last) = last {
                if let Some(wire) = graph.wires().iter().find(|w| w.from.node == last && w.to.node == start).cloned() {
                    graph.disconnect(&wire);
                }
                let _ = graph.connect(&reg, PortRef::new(last, SETUP), PortRef::new(id, SETUP));
            }
            let _ = graph.connect(&reg, PortRef::new(id, SETUP), PortRef::new(start, SETUP));
        }
    }
    created
}

// ---- the training plan ----------------------------------------------------------

/// Where the files a training run needs were written.
pub struct PlanFiles<'a> {
    /// A Text Dataset's file (its path inside the `.magic` file) on disk.
    pub dataset: &'a dyn Fn(&str) -> Option<String>,
    /// The weights to start from, on disk.
    pub weights: Option<String>,
    /// The prepared pictures (vision).
    pub pictures: Option<String>,
    pub out_dir: String,
    /// CUDA devices to train on; empty: the CPU.
    pub devices: Vec<u32>,
}

/// A Hugging Face Dataset block's entries in the plan: its corpus, then what is wired into its
/// With input – a Text Dataset's files, or another Hugging Face Dataset with its own.
fn corpus_entries(graph: &Graph, node: NodeId, files: &PlanFiles<'_>, depth: usize) -> Result<Vec<Value>, String> {
    let block = graph.node(node).ok_or("a dataset block is missing")?;
    let corpus: CorpusParams = block.params_or_default();
    if corpus.repo.trim().is_empty() {
        return Err("The Hugging Face Dataset block needs a dataset (owner/name).".into());
    }
    let mut entries = vec![json!({ "corpus": corpus })];
    let Some(with) = graph.data_source(node, "with") else { return Ok(entries) };
    let Some(source) = graph.node(with.node) else { return Ok(entries) };
    if source.kind == kinds::CORPUS {
        if depth > 16 {
            return Err("Hugging Face Dataset blocks are wired in a circle.".into());
        }
        entries.extend(corpus_entries(graph, with.node, files, depth + 1)?);
        return Ok(entries);
    }
    let params: TextDataParams = source.params_or_default();
    let list = if with.port == "negative" { &params.negative } else { &params.positive };
    for file in list {
        entries
            .push(Value::from((files.dataset)(file).ok_or_else(|| format!("The dataset file “{file}” is missing."))?));
    }
    Ok(entries)
}

/// The training plan the trainer (`runtime/python/magi_train`) reads.
pub fn plan(pipeline: &Pipeline, graph: &Graph, model: &MagicModel, files: &PlanFiles<'_>) -> Result<Value, String> {
    let kind = pipeline.kind(model);
    let mut datasets = Map::new();
    // A dataset output with no files adds nothing.
    let mut key_of = |(node, port): &(NodeId, String)| -> Result<Option<String>, String> {
        let block = graph.node(*node).ok_or("a dataset block is missing")?;
        if block.kind == kinds::CORPUS {
            // Read where the training runs: the plan names it, the trainer fetches it.
            let key = format!("{node}.{port}");
            datasets.insert(key.clone(), Value::Array(corpus_entries(graph, *node, files, 0)?));
            return Ok(Some(key));
        }
        let params: TextDataParams = block.params_or_default();
        let list = if port == "negative" { &params.negative } else { &params.positive };
        if list.is_empty() {
            return Ok(None);
        }
        let key = format!("{node}.{port}");
        let mut paths = Vec::new();
        for file in list {
            paths.push(Value::from(
                (files.dataset)(file).ok_or_else(|| format!("The dataset file “{file}” is missing."))?,
            ));
        }
        datasets.insert(key.clone(), Value::Array(paths));
        Ok(Some(key))
    };
    let mut stages = Vec::new();
    for stage in &pipeline.stages {
        let data: Vec<String> =
            stage.data.iter().map(&mut key_of).collect::<Result<Vec<_>, _>>()?.into_iter().flatten().collect();
        let negatives: Vec<String> =
            stage.negatives.iter().map(&mut key_of).collect::<Result<Vec<_>, _>>()?.into_iter().flatten().collect();
        let mut entry = stage.params.clone();
        entry.remove("note");
        let ty = match stage.kind.as_str() {
            kinds::PRETRAIN => "pretrain",
            kinds::FINETUNE => {
                let p: FinetuneParams = serde_json::from_value(Value::Object(stage.params.clone())).unwrap_or_default();
                entry.insert("adapter".into(), Value::from(p.method));
                "sft"
            }
            kinds::RL => {
                let p: RlParams = serde_json::from_value(Value::Object(stage.params.clone())).unwrap_or_default();
                let rules: Vec<Value> = p
                    .rules
                    .iter()
                    .map(|r| json!({ "kind": r.kind, "text": r.text, "score": r.score, "tokens": r.tokens, "ngram": r.tokens, "repeats": r.repeats }))
                    .collect();
                entry.insert("rules".into(), Value::Array(rules));
                "rl"
            }
            kinds::COMPACTION => "compaction",
            kinds::VISION_TRAIN => "vision",
            _ => continue,
        };
        let name = registry().get(&stage.kind).map(|d| d.name.clone()).unwrap_or_default();
        if kind == ModelKind::Text && ty != "vision" && data.is_empty() {
            return Err(format!("{name} has nothing to learn from: wire a Text Dataset with files into it."));
        }
        if kind == ModelKind::Vision && ty != "vision" {
            return Err(format!("{name} is for text models; a vision model trains with Vision Training."));
        }
        if kind == ModelKind::Text && ty == "vision" {
            return Err("Vision Training is for vision models.".into());
        }
        let mut optimizer = serde_json::to_value(&stage.optimizer).map_err(|e| e.to_string())?;
        if let Some(o) = optimizer.as_object_mut() {
            o.remove("note");
        }
        entry.insert("type".into(), Value::from(ty));
        entry.insert("node".into(), Value::from(stage.node));
        // The block's own name, for the console's lines about it.
        if let Some(label) = graph.node(stage.node).map(|n| n.label.trim()).filter(|l| !l.is_empty()) {
            entry.insert("name".into(), Value::from(label));
        }
        entry.insert("data".into(), json!(data));
        entry.insert("negatives".into(), json!(negatives));
        entry.insert("optimizer".into(), optimizer);
        stages.push(Value::Object(entry));
    }
    let mut architecture = serde_json::to_value(&pipeline.architecture).map_err(|e| e.to_string())?;
    if let Some(a) = architecture.as_object_mut() {
        a.remove("blocks");
    }
    let mut plan = json!({
        "kind": if kind == ModelKind::Text { "text" } else { "vision" },
        "out_dir": files.out_dir,
        "seed": 0,
        "architecture": architecture,
        "start": { "mode": pipeline.entry.kind.trim_start_matches("magic."), "weights": files.weights },
        "datasets": datasets,
        "stages": stages,
        "devices": files.devices,
    });
    match kind {
        ModelKind::Text => {
            if pipeline.architecture.layers.is_empty() {
                return Err(
                    "The model has no layers: add Transformer Layers (or another layer block) after Model Size.".into(),
                );
            }
            let mut text = model.text.clone().unwrap_or_default();
            text.context = pipeline.architecture.context;
            text.ensure_specials();
            plan["text"] = serde_json::to_value(text).map_err(|e| e.to_string())?;
            // The native engine runs most text models from their weights: an ONNX model is made only
            // when it cannot, or when Compile Package asks for an ONNX file.
            let asked = pipeline.compile.as_ref().is_some_and(|(_, c)| c.onnx);
            plan["onnx"] = json!(asked || pipeline.architecture.native_problem().is_some());
        }
        ModelKind::Vision => {
            let pictures = files
                .pictures
                .clone()
                .ok_or("The model has no pictures to learn from: add some in the Vision Dataset block.")?;
            if model.classes.len() < 2 {
                return Err("A vision model needs at least two answers (the Answers block).".into());
            }
            plan["vision"] = json!({ "pictures": pictures, "classes": model.classes.len() });
        }
    }
    Ok(plan)
}

// ---- text datasets ----------------------------------------------------------

/// Whether a dataset file holds one JSON sample per line (`.jsonl`), rather
/// than plain text split at blank lines (`.md`, `.txt`).
pub fn is_jsonl(file: &str) -> bool {
    let lower = file.to_lowercase();
    lower.ends_with(".jsonl") || lower.ends_with(".json")
}

/// The samples of a dataset file: its lines (`.jsonl`) or its paragraphs.
pub fn split_samples(file: &str, content: &str) -> Vec<String> {
    if is_jsonl(file) {
        content.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_owned).collect()
    } else {
        content
            .replace("\r\n", "\n")
            .split("\n\n")
            .map(str::trim)
            .filter(|b| !b.is_empty())
            .map(str::to_owned)
            .collect()
    }
}

/// A dataset file from its samples.
pub fn join_samples(file: &str, samples: &[String]) -> String {
    let separator = if is_jsonl(file) { "\n" } else { "\n\n" };
    let mut out = samples.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(separator);
    out.push('\n');
    out
}

/// What kind of sample one is, or what is wrong with it: the JSON, the
/// fields the trainer reads, and every tool call in it.
pub fn check_sample(jsonl: bool, sample: &str, text: &TextSpec) -> Result<&'static str, String> {
    if !jsonl {
        check_tools(sample, text, None)?;
        return Ok("Text");
    }
    let value: Value = serde_json::from_str(sample).map_err(|e| format!("Not valid JSON: {e}"))?;
    let Value::Object(item) = &value else {
        return Err("A sample is a JSON object, such as {\"prompt\": …, \"response\": …}.".into());
    };
    let text_of = |key: &str| item.get(key).map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_owned));
    // The tools a sample offers, by name; without “tools”, the model's.
    let offered: Option<Vec<String>> = match item.get("tools") {
        None => None,
        Some(Value::Array(tools)) => Some(
            tools
                .iter()
                .map(|tool| {
                    tool.get("function")
                        .unwrap_or(tool)
                        .get("name")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                        .ok_or_else(|| "Each tool in “tools” needs a “name”.".to_owned())
                })
                .collect::<Result<_, _>>()?,
        ),
        Some(_) => return Err("“tools” must be a list of tools.".into()),
    };
    if let Some(effort) = item.get("effort") {
        let named = match effort {
            Value::String(level) => vec![level.as_str()],
            Value::Array(levels) if levels.iter().all(Value::is_string) => {
                levels.iter().filter_map(Value::as_str).collect()
            }
            _ => return Err("“effort” is a level's name, such as \"superior\", or a list of them.".into()),
        };
        if let Some(levels) = &text.effort
            && let Some(other) = named.iter().find(|effort| !levels.levels.iter().any(|l| l == *effort))
        {
            return Err(format!(
                "“effort” names “{other}”, which is not one of the model's levels ({}).",
                levels.levels.join(", ")
            ));
        }
    }
    let check = |s: &str| check_tools(s, text, offered.as_deref());
    if let Some(messages) = item.get("messages") {
        let Value::Array(messages) = messages else { return Err("“messages” must be a list.".into()) };
        for message in messages {
            let role = message.get("role").and_then(Value::as_str).unwrap_or_default();
            if !["system", "user", "assistant", "tool"].contains(&role) {
                return Err(format!("A message's role must be system, user, assistant or tool, not “{role}”."));
            }
            match message.get("content") {
                Some(Value::String(content)) => check(content)?,
                None | Some(Value::Null) if role == "assistant" => {}
                Some(other) if role == "tool" => check(&other.to_string())?,
                _ => return Err(format!("A {role} message needs its “content” as text.")),
            }
            if let Some(thinking) = message.get("thinking")
                && !thinking.is_string()
            {
                return Err("“thinking” is text.".into());
            }
            // A turn marked "learn": false is read, not learned: a wrong call the model then recovers
            // from may name a tool it does not have.
            let learned = message.get("learn").and_then(Value::as_bool) != Some(false);
            if let Some(calls) = message.get("tool_calls") {
                let Value::Array(calls) = calls else { return Err("“tool_calls” must be a list.".into()) };
                let any_tool = TextSpec::default();
                for call in calls {
                    let call = call.get("function").unwrap_or(call);
                    if learned {
                        check_call(call, text, offered.as_deref())?;
                    } else {
                        check_call(call, &any_tool, None)?;
                    }
                }
            }
        }
        if !messages.iter().any(|m| m.get("role").and_then(Value::as_str) == Some("assistant")) {
            return Err("A conversation needs an answer: a message with the role “assistant”.".into());
        }
        let tools = messages.iter().any(|m| m.get("tool_calls").is_some() || m.get("role") == Some(&json!("tool")));
        return Ok(if tools { "Tool use" } else { "Conversation" });
    }
    if let (Some(context), Some(compact)) = (text_of("context"), text_of("compact")) {
        check(&context)?;
        check(&compact)?;
        return Ok("Compaction");
    }
    if let Some(prompt) = text_of("prompt") {
        check(&prompt)?;
        if text_of("rejected").or_else(|| text_of("bad")).is_some() {
            // An answer not to give may be broken on purpose: only the request is checked.
            return Ok("Not to answer");
        }
        let response =
            text_of("response").ok_or("A request needs its “response” (or “rejected” for an answer not to give).")?;
        if prompt.trim().is_empty() || response.trim().is_empty() {
            return Err("The request or its answer is empty: there is nothing to learn from it.".into());
        }
        check(&response)?;
        return Ok(if item.get("thinking").is_some_and(|t| t.as_str().is_some_and(|t| !t.is_empty())) {
            "Reasoning"
        } else {
            "Request and answer"
        });
    }
    if let Some(plain) = text_of("text") {
        check(&plain)?;
        return Ok("Text");
    }
    Err("Not a sample the trainer reads: give prompt and response, messages, context and compact, or text.".into())
}

/// A tool call as JSON: a name the sample (or the model) offers, and its arguments.
fn check_call(call: &Value, spec: &TextSpec, offered: Option<&[String]>) -> Result<(), String> {
    let name = call.get("name").and_then(Value::as_str).ok_or("A tool call needs a “name”.")?;
    let arguments = call.get("arguments").or_else(|| call.get("args"));
    if arguments.is_some_and(|a| !a.is_object() && !a.is_string()) {
        return Err("A tool call's “arguments” must be an object.".into());
    }
    match offered {
        Some(names) if !names.iter().any(|n| n == name) => {
            Err(format!("The tool “{name}” is not one of the sample's “tools”."))
        }
        None if !spec.tools.is_empty() && !spec.tools.iter().any(|t| t.name == name) => {
            Err(format!("The tool “{name}” is not one of the model's tools (the Tools block)."))
        }
        _ => Ok(()),
    }
}

/// Every tool call in a text must be written the one way:
/// `<tool_call>{"name": …, "arguments": {…}}</tool_call>`.
fn check_tools(text: &str, spec: &TextSpec, offered: Option<&[String]>) -> Result<(), String> {
    let (open, close) = (spec.tool_tags.0.as_str(), spec.tool_tags.1.as_str());
    let mut rest = text;
    while let Some(start) = rest.find(open) {
        let after = &rest[start + open.len()..];
        let end = after.find(close).ok_or_else(|| format!("A tool call is not closed with {close}."))?;
        let call: Value = serde_json::from_str(after[..end].trim())
            .map_err(|_| format!("A tool call is not JSON: {}", after[..end].trim()))?;
        check_call(&call, spec, offered)?;
        rest = &after[end + close.len()..];
    }
    if rest.contains(close) {
        return Err(format!("{close} without {open}."));
    }
    Ok(())
}

// ---- the example workspace ------------------------------------------------------

/// The system prompt of the example text model.
pub const EXAMPLE_SYSTEM: &str = "You are Magi, an agent made in MacroMagi. You know how to work, not facts or exact code: \
think, plan in steps, look up what you need in MagiDB (db_search, db_read) or on the web, act with your tools, check the \
results, and answer briefly.";

/// The tools a Magi model has, in the order its datasets list them (examples/make_work_data.py):
/// MacroMagi's native tools for an agent's work.
pub const MAGI_TOOLS: [&str; 16] = [
    "db_search",
    "db_read",
    "web_search",
    "fetch_page",
    "list_files",
    "read_file",
    "search_files",
    "write_file",
    "edit_file",
    "run_command",
    "open_document",
    "edit_document",
    "view_image",
    "calculator",
    "get_time",
    "get_weather",
];

/// A Magi model's tools: [`MAGI_TOOLS`], as the native tools describe themselves.
pub fn example_tools() -> Vec<ToolSpec> {
    let all = crate::native_tools::built_in();
    MAGI_TOOLS.iter().filter_map(|name| all.iter().find(|t| t.name == *name)).map(|t| t.spec()).collect()
}

/// The example datasets a new Magi model starts with: written for MacroMagi by
/// `examples/make_examples.py`, `make_agent_data.py` and `make_work_data.py` (plain English,
/// conversation, tools one at a time and many in a row, long pieces of work – code, research,
/// documents, pictures – with MagiDB, compaction, answers not to give), and taken from open
/// datasets by `examples/open_data.py` (conversations, tool calls; credits in
/// `examples/SOURCES.md`). Paths are inside the `.magic` file; `true` marks “what not to do”.
pub fn example_datasets() -> Vec<(String, String, bool)> {
    let file = |name: &str, content: &str, negative: bool| (format!("data/text/{name}"), content.to_owned(), negative);
    vec![
        file("language.md", include_str!("examples/language.md"), false),
        // Long pieces of work: tens of tool calls, MagiDB first (examples/make_work_data.py).
        file("work.jsonl", include_str!("examples/work.jsonl"), false),
        file("chat.jsonl", include_str!("examples/chat.jsonl"), false),
        file("tools.jsonl", include_str!("examples/tools.jsonl"), false),
        file("titles.jsonl", include_str!("examples/titles.jsonl"), false),
        file("knowing.jsonl", include_str!("examples/knowing.jsonl"), false),
        file("simple_qa.jsonl", include_str!("examples/simple_qa.jsonl"), false),
        file("compaction.jsonl", include_str!("examples/compaction.jsonl"), false),
        // Written by examples/make_agent_data.py: a router's habits.
        file("thinking.jsonl", include_str!("examples/thinking.jsonl"), false),
        file("agent.jsonl", include_str!("examples/agent.jsonl"), false),
        file("small_talk.jsonl", include_str!("examples/small_talk.jsonl"), false),
        file("explain.jsonl", include_str!("examples/explain.jsonl"), false),
        // Taken from open datasets by examples/open_data.py (credits in examples/SOURCES.md).
        file("conversations.jsonl", include_str!("examples/conversations.jsonl"), false),
        file("functions.jsonl", include_str!("examples/functions.jsonl"), false),
        file("avoid.jsonl", include_str!("examples/avoid.jsonl"), true),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn every_kind() -> Vec<&'static str> {
        use kinds::*;
        let mut all: Vec<&str> =
            vec![ZERO, BASE, CONFIG, MOE, PERSONA, TEXT_DATA, CORPUS, IMAGE_DATA, OPTIMIZER, COMPILE, PROMPT, TEST];
        all.extend(SETUP_BLOCKS);
        all.extend(ARCHITECTURE);
        all.extend(STAGES);
        all
    }

    #[test]
    fn untouched_old_optimizers_take_the_tuned_rates() {
        let registry = registry();
        let mut graph = Graph::default();
        let old = graph.add(&registry, kinds::OPTIMIZER, [0.0, 0.0]).unwrap();
        let changed = graph.add(&registry, kinds::OPTIMIZER, [0.0, 100.0]).unwrap();
        for (id, muon) in [(old, 0.02), (changed, 0.01)] {
            let node = graph.node_mut(id).unwrap();
            node.params.remove("embed_lr");
            node.set_param("muon_lr", json!(muon));
        }
        upgrade(&mut graph);
        let rates = |id| {
            let p: OptimizerParams = graph.node(id).unwrap().params_or_default();
            (p.muon_lr, p.embed_lr)
        };
        assert_eq!(rates(old), (0.004, 0.02));
        assert_eq!(rates(changed), (0.01, 0.02));
        assert!(!graph.node(changed).unwrap().params.contains_key("embed_lr"));
    }

    #[test]
    fn every_magic_block_has_complete_defaults() {
        let registry = registry();
        for id in every_kind() {
            assert!(registry.get(id).is_some(), "{id}");
        }
        assert_eq!(registry.len(), every_kind().len(), "every block is listed");
        assert!(Registry::builtin().get(kinds::ZERO).is_none(), "macros never see them");
    }

    fn text_model() -> MagicModel {
        MagicModel {
            kind: ModelKind::Text,
            text: Some(TextSpec {
                thinking: Some(ThinkingSpec::default()),
                effort: Some(magi_effort()),
                tools: example_tools(),
                ..TextSpec::default()
            }),
            name: "Helper".into(),
            ..MagicModel::default()
        }
    }

    fn files() -> Vec<(String, bool)> {
        example_datasets().into_iter().map(|(f, _, n)| (f, n)).collect()
    }

    fn plan_files<'a>(dataset: &'a dyn Fn(&str) -> Option<String>) -> PlanFiles<'a> {
        PlanFiles { dataset, weights: None, pictures: None, out_dir: "/tmp/out".into(), devices: vec![] }
    }

    #[test]
    fn a_new_text_model_gets_a_working_pipeline() {
        let mut graph = Graph::from(default_canvas(&text_model(), false, false, &files()));
        let found = pipeline(&graph).unwrap();
        assert_eq!(found.entry.kind, kinds::ZERO);
        let stages: Vec<&str> = found.stages.iter().map(|s| s.kind.as_str()).collect();
        assert_eq!(stages, [kinds::PRETRAIN, kinds::FINETUNE, kinds::COMPACTION, kinds::RL], "Magi's recipe");
        assert!(found.stages.iter().all(|s| s.has_optimizer), "an optimizer is wired into every stage");
        let optimizers: Vec<&str> = found.stages.iter().map(|s| s.optimizer.name.as_str()).collect();
        assert_eq!(optimizers, ["muon", "muon", "muon", "adamw"], "reinforcement is gentler");
        let a = &found.architecture;
        assert_eq!(a.rope_scaling.as_ref().map(|s| (s.original, s.target)), Some((8192, 131072)), "YaRN to 128K");
        assert_eq!((a.width, a.heads, a.kv_heads, a.layers.len()), (768, 12, 4, 32));
        assert!(a.tied && a.mtp.is_some());
        // Three plain layers, then experts; three of four layers have a window of 1,024-token blocks.
        let ffn: Vec<&str> = a.layers.iter().map(|l| l["ffn"]["kind"].as_str().unwrap_or("dense")).collect();
        assert_eq!(&ffn[..4], ["dense", "dense", "dense", "moe"]);
        assert!(ffn[3..].iter().all(|k| *k == "moe"));
        assert_eq!(a.layers[3]["ffn"]["experts"], 16);
        let windows: Vec<u64> = a.layers.iter().map(|l| l["attention"]["window"].as_u64().unwrap_or(0)).collect();
        assert_eq!(&windows[..8], [1024, 1024, 1024, 0, 1024, 1024, 1024, 0]);
        assert!(found.compile.is_some());

        // The setup chain makes the model what it says.
        let mut model = MagicModel::default();
        derive(&mut graph, &mut model, false);
        assert_eq!((model.kind, model.name.as_str()), (ModelKind::Text, "Helper"));
        let text = model.text.as_ref().unwrap();
        assert_eq!(text.context, 8192);
        assert!(text.thinking.is_some() && text.effort.is_some(), "Thinking and Reasoning Effort are chained in");
        // Unchaining Thinking switches it off.
        let thinking = graph.nodes().find(|n| n.kind == kinds::THINKING).unwrap().id;
        graph.node_mut(thinking).unwrap().enabled = false;
        derive(&mut graph, &mut model, false);
        assert!(model.text.as_ref().unwrap().thinking.is_none());
        graph.node_mut(thinking).unwrap().enabled = true;
        // Unchaining Model Info leaves the model unnamed, and the block says why.
        let info = graph.nodes().find(|n| n.kind == kinds::INFO).unwrap().id;
        let wire = graph.wires().iter().find(|w| w.from.node == info).cloned().unwrap();
        graph.disconnect(&wire);
        derive(&mut graph, &mut model, false);
        assert_eq!(model.name, "");
        assert!(graph.node(info).unwrap().param::<String>("note").unwrap().contains("Setup"));

        let found = pipeline(&graph).unwrap();
        let dataset = |file: &str| Some(format!("/tmp/{}", file.rsplit('/').next().unwrap()));
        let plan = plan(&found, &graph, &model, &plan_files(&dataset)).unwrap();
        assert_eq!(plan["kind"], "text");
        let types: Vec<&str> = plan["stages"].as_array().unwrap().iter().map(|s| s["type"].as_str().unwrap()).collect();
        assert_eq!(types, ["pretrain", "sft", "compaction", "rl"]);
        let entries = |stage: usize| plan["datasets"][plan["stages"][stage]["data"][0].as_str().unwrap()].clone();
        // Pretraining: FineWeb-Edu, the code corpus, then the example files of plain text.
        let pretraining = entries(0);
        assert_eq!(pretraining[0]["corpus"]["repo"], "HuggingFaceFW/fineweb-edu");
        assert_eq!(pretraining[1]["corpus"]["repo"], "OpenCoder-LLM/opc-fineweb-code-corpus");
        assert!(pretraining.as_array().unwrap().iter().any(|f| f.as_str().is_some_and(|f| f.ends_with("language.md"))));
        // Fine-tuning: smol-smoltalk's conversations, and the agent work.
        let tuning = entries(1);
        assert_eq!(tuning[0]["corpus"]["field"], "messages");
        let tuned =
            |name: &str| tuning.as_array().unwrap().iter().any(|f| f.as_str().is_some_and(|f| f.ends_with(name)));
        assert!(tuned("work.jsonl") && tuned("tools.jsonl") && tuned("compaction.jsonl"));
        assert_eq!(plan["stages"][3]["rules"].as_array().unwrap().len(), 7, "the agent's reward rules");
        assert_eq!(
            plan["architecture"]["layers"][0]["attention"],
            json!({ "kind": "full", "qk_norm": true, "window": 1024 })
        );
        assert_eq!(plan["text"]["effort"]["default"], "standard");
        assert!(plan["architecture"].get("blocks").is_none());
        let key = plan["stages"][1]["negatives"][0].as_str().unwrap().to_owned();
        assert_eq!(plan["datasets"][&key][0], "/tmp/avoid.jsonl");
    }

    #[test]
    fn the_pipeline_explains_what_is_wrong() {
        // A model made elsewhere is only used as it is, with its setup on the canvas.
        let vision = MagicModel { classes: vec![Default::default(); 2], ..MagicModel::default() };
        let mut graph = Graph::from(default_canvas(&vision, false, true, &[]));
        assert!(!pipeline(&graph).unwrap().trains());
        let mut model = vision.clone();
        derive(&mut graph, &mut model, true);
        assert_eq!(model.classes.len(), 2, "the Answers block keeps them");

        // Training stages after Use As It Is are refused, with a reason.
        let config = graph.nodes().find(|n| n.kind == kinds::CONFIG).unwrap().id;
        let compile = graph.nodes().find(|n| n.kind == kinds::COMPILE).unwrap().id;
        let wire = graph.wires().iter().find(|w| w.from.node == config && w.to.node == compile).cloned().unwrap();
        graph.disconnect(&wire);
        let stage = graph.add_with(kinds::VISION_TRAIN, [0.0, 0.0], Map::new());
        graph.connect(&registry(), PortRef::new(config, MODEL), PortRef::new(stage, MODEL)).unwrap();
        assert!(pipeline(&graph).unwrap_err().contains("does not train"));

        // Two starts.
        let mut two = Graph::from(default_canvas(&text_model(), false, false, &files()));
        two.add_with(kinds::ZERO, [0.0, 0.0], Map::new());
        assert!(pipeline(&two).unwrap_err().contains("more than one"));

        // A stage without its datasets.
        let mut bare = Graph::from(default_canvas(&text_model(), false, false, &files()));
        let data: Vec<NodeId> =
            bare.nodes().filter(|n| n.kind == kinds::TEXT_DATA || n.kind == kinds::CORPUS).map(|n| n.id).collect();
        for id in data {
            bare.remove(id);
        }
        let found = pipeline(&bare).unwrap();
        let none = |_: &str| None;
        assert!(plan(&found, &bare, &text_model(), &plan_files(&none)).unwrap_err().contains("nothing to learn from"));

        // Heads that do not divide the width.
        let mut odd = Graph::from(default_canvas(&text_model(), false, false, &files()));
        let size = odd.nodes().find(|n| n.kind == kinds::SIZE).unwrap().id;
        odd.node_mut(size).unwrap().set_param("heads", 5);
        assert!(pipeline(&odd).unwrap_err().contains("divide"));
    }

    #[test]
    fn commons_put_magi_back_in_place() {
        assert_eq!(commons().iter().map(|c| c.name).collect::<Vec<_>>(), ["Magi", "Magi 150M"], "Magi only");
        let mut graph = Graph::from(default_canvas(&text_model(), false, false, &files()));
        // Changed by hand…
        let layers: Vec<NodeId> = graph.nodes().filter(|n| n.kind == kinds::LAYERS).map(|n| n.id).collect();
        graph.node_mut(layers[1]).unwrap().set_param("layers", 2);
        assert_eq!(pipeline(&graph).unwrap().architecture.layers.len(), 5);
        // …and put back: the stages stay wired after it, and the experts block is not doubled.
        apply_common(&mut graph, &commons()[0], [0.0, 0.0]);
        let found = pipeline(&graph).unwrap();
        assert_eq!(found.stages.len(), 4);
        assert_eq!(found.architecture.layers.len(), 32);
        assert_eq!(graph.nodes().filter(|n| n.kind == kinds::MOE).count(), 1);
        let mut model = MagicModel { text: Some(TextSpec::default()), ..text_model() };
        derive(&mut graph, &mut model, false);
        let text = model.text.as_ref().unwrap();
        assert!(text.thinking.as_ref().is_some_and(|t| t.on_by_default), "Magi thinks before each step");
        assert!(text.effort.is_some());
    }

    #[test]
    fn a_trained_model_opens_with_its_own_architecture() {
        let graph = Graph::from(default_canvas(&text_model(), false, false, &files()));
        let trained = MagicModel { architecture: Some(pipeline(&graph).unwrap().architecture), ..text_model() };
        let reopened = Graph::from(default_canvas(&trained, true, true, &files()));
        let found = pipeline(&reopened).unwrap();
        assert_eq!(found.entry.kind, kinds::BASE);
        assert_eq!(found.architecture.layers, trained.architecture.as_ref().unwrap().layers);
        let stages: Vec<&str> = found.stages.iter().map(|s| s.kind.as_str()).collect();
        assert_eq!(stages, [kinds::FINETUNE, kinds::COMPACTION, kinds::RL], "no pretraining from trained weights");
    }

    #[test]
    fn dataset_samples_are_checked() {
        let spec = TextSpec { effort: Some(EffortSpec::default()), ..TextSpec::default() };
        let check = |sample: &str| check_sample(true, sample, &spec);
        assert_eq!(check(r#"{"prompt": "hi", "response": "yo"}"#), Ok("Request and answer"));
        assert_eq!(
            check(r#"{"prompt": "hi", "thinking": "easy", "response": "yo", "effort": "low"}"#),
            Ok("Reasoning")
        );
        assert!(check(r#"{"prompt": "hi", "response": "yo", "effort": "extreme"}"#).unwrap_err().contains("levels"));
        assert_eq!(check(r#"{"prompt": "hi", "rejected": "no"}"#), Ok("Not to answer"));
        assert_eq!(check(r#"{"context": "long", "compact": "short"}"#), Ok("Compaction"));
        assert_eq!(check(r#"{"text": "plain"}"#), Ok("Text"));
        let tool = r#"{"prompt": "go", "response": "<tool_call>{\"name\": \"press\", \"arguments\": {\"key\": \"A\"}}</tool_call>"}"#;
        assert_eq!(check(tool), Ok("Request and answer"));
        let conversation = r#"{"tools": [{"name": "web_search"}], "messages": [{"role": "user", "content": "news?"},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "web_search", "arguments": {"query": "news"}}]},
            {"role": "tool", "content": "rain"}, {"role": "assistant", "content": "It rains."}]}"#
            .replace('\n', " ");
        assert_eq!(check(&conversation), Ok("Tool use"));
        assert!(
            check(
                &conversation.replace("\"name\": \"web_search\", \"arguments\"", "\"name\": \"other\", \"arguments\"")
            )
            .unwrap_err()
            .contains("“tools”")
        );
        assert!(check("{not json").is_err());
        assert!(check(r#"{"prompt": "hi"}"#).unwrap_err().contains("response"));
        assert!(check(r#"{"messages": [{"role": "user", "content": "hi"}]}"#).unwrap_err().contains("assistant"));
        assert!(
            check(r#"{"prompt": "go", "response": "<tool_call>{\"arguments\": {}}</tool_call>"}"#)
                .unwrap_err()
                .contains("name")
        );
        assert!(
            check(r#"{"prompt": "go", "response": "<tool_call>{\"name\": \"x\"}"}"#)
                .unwrap_err()
                .contains("not closed")
        );
        assert!(check_sample(false, "a <tool_call>oops</tool_call>", &spec).is_err());

        let samples = split_samples("a.md", "one\r\n\r\ntwo\n\n\nthree");
        assert_eq!(samples, ["one", "two", "three"]);
        assert_eq!(join_samples("a.md", &samples), "one\n\ntwo\n\nthree\n");
        assert_eq!(split_samples("a.jsonl", "{}\n\n{}\n"), ["{}", "{}"]);
    }

    #[test]
    fn the_examples_are_valid_samples() {
        let spec = text_model().text.unwrap();
        for (file, content, _) in example_datasets() {
            let samples = split_samples(&file, &content);
            assert!(samples.len() >= 5, "{file}");
            for sample in &samples {
                let checked = check_sample(is_jsonl(&file), sample, &spec);
                assert!(checked.is_ok(), "{file}: {sample}: {checked:?}");
            }
        }
        let kinds: BTreeSet<&str> = example_datasets()
            .iter()
            .filter(|(f, ..)| is_jsonl(f))
            .flat_map(|(f, content, _)| split_samples(f, content).into_iter().map(move |s| (f.clone(), s)))
            .filter_map(|(f, s)| check_sample(is_jsonl(&f), &s, &spec).ok())
            .collect();
        assert_eq!(
            kinds,
            BTreeSet::from([
                "Compaction",
                "Conversation",
                "Not to answer",
                "Reasoning",
                "Request and answer",
                "Tool use"
            ]),
            "every kind of sample is shown"
        );
    }
}

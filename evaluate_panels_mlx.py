"""
evaluate_panels_mlx.py
 
Standalone evaluation supporting two methods:
  --method panels   Video Panels baseline (uniform sampling, 2×2 grid)
  --method ours     Our method (CLIP-scored selection, 2×2 grid)
 
Usage:
    python evaluate_panels_mlx.py --limit 5
    python evaluate_panels_mlx.py --limit 50 --gamma 1.0
    python evaluate_panels_mlx.py --limit 5 --method ours
    python evaluate_panels_mlx.py --limit 1 --C 4          # small memory test
    python evaluate_panels_mlx.py --limit 1 --C 32         # real target config
    python evaluate_panels_mlx.py --limit 5 --model mlx-community/Qwen2-VL-2B-Instruct-4bit
    python evaluate_panels_mlx.py --limit 5 --save-panels  # keep panel PNGs
    python evaluate_panels_mlx.py --duration medium --limit 10 --method ours
"""
 
import argparse
import json
import os
import re
import tempfile
 
from datasets import load_dataset
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
 
from video_panels_baseline import PanelConfig, video_panels_pipeline
 
# our_method.py is optional - only needed for --method ours (CLIP-scored
# selection). Import defensively so the Video Panels baseline and the
# --no-visual diagnostic both run without it present.
try:
    from our_method import OurMethodConfig, our_method_pipeline, ClipScorer
except ImportError:
    OurMethodConfig = our_method_pipeline = ClipScorer = None
 
 
# ---- defaults - now overridable from the command line, see main() ----
DEFAULT_MLX_MODEL_ID = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
DEFAULT_C = 32
VIDEO_DATA_DIR = "/Volumes/TRANSCEND/hf_cache/videomme/data"  # your already-working path
HF_CACHE_DIR = "/Volumes/TRANSCEND/hf_cache/videomme"
PANELS_OUTPUT_DIR = "saved_panels"  # where kept panels land when --save-panels is used
 
 
def build_prompt(question: str, options: list) -> str:
    """Matches the Video Panels paper's prompt exactly.
 
    From their implementation details: "The VLMs are given the question
    and the multiple-choice answers in the format expected by the model,
    after which all VLMs get the standard instruction to 'Answer with
    the option's letter from the given choices directly.\\n'"
 
    Deliberately minimal - no extra "do not explain" instructions, since
    their Table 7 shows prompt variations move Qwen2.5-VL by ~1 point on
    VMME, and matching their setup exactly removes prompt phrasing as an
    uncontrolled variable in the comparison."""
    option_text = "\n".join(options)
    return f"{question}\n{option_text}\nAnswer with the option's letter from the given choices directly.\n"
 
 
def extract_letter(text: str) -> str:
    """Grabs the FIRST standalone A/B/C/D letter in the text. This is
    correct (not the earlier 'take the last letter' logic) because the
    model's real answer is emitted immediately, followed by an
    end-of-text token - anything after that point is generation
    continuing past where it should have stopped, and should be
    ignored entirely, not searched for a 'better' final letter."""
    matches = re.findall(r"\b([A-D])\b", text.strip())
    return matches[0] if matches else text.strip()[:1].upper()
 
 
def truncate_at_stop_token(text: str) -> str:
    """Cuts the response off at the first end-of-text marker, since
    mlx_vlm's generate() was observed NOT reliably stopping generation
    there itself - everything after this point in a raw response is
    the model continuing to sample past where it already finished,
    producing unrelated content (seen directly in testing: real
    answers followed by a totally different, unrelated question about
    oceans). Truncating here BEFORE letter-extraction is the actual
    fix - taking the first letter alone isn't fully safe if an early
    stop-token boundary isn't respected first."""
    for stop_marker in ("<|endoftext|>", "<|im_end|>", "</s>"):
        if stop_marker in text:
            text = text.split(stop_marker)[0]
    return text
 
 
def extract_text(raw_output) -> str:
    """generate()'s return type may be a plain string, or a structured
    result object (e.g. with a .text attribute) depending on mlx_vlm
    version - handle both rather than assuming str(raw_output) is
    already clean, which could otherwise stringify an entire object
    repr (tokens, metadata, etc.) instead of just the answer text."""
    if isinstance(raw_output, str):
        return raw_output
    for attr in ("text", "content", "output"):
        if hasattr(raw_output, attr):
            return str(getattr(raw_output, attr))
    return str(raw_output)
 
 
def resolve_video_path(video_id: str) -> str:
    for ext in ("mp4", "MP4", "mkv"):
        candidate = os.path.join(VIDEO_DATA_DIR, f"{video_id}.{ext}")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No video file found for videoID={video_id} in {VIDEO_DATA_DIR}")
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="number of questions to evaluate. Omit (or leave unset) "
                              "to run the full dataset. Use small values (5, 10, 50) "
                              "during development/debugging.")
    parser.add_argument("--gamma", type=float, default=1.0, help="paneling spacing constraint (xfps)")
    parser.add_argument("--output", default="results_panels_mlx.jsonl")
    parser.add_argument("--C", type=int, default=DEFAULT_C,
                         help="frame budget (panels sent to the model). Start small "
                              "(e.g. 4) to test memory headroom, then step up toward "
                              "the real target value (32 for Qwen2.5VL-7B medium-context).")
    parser.add_argument("--model", default=DEFAULT_MLX_MODEL_ID,
                         help="MLX model ID, e.g. mlx-community/Qwen2-VL-2B-Instruct-4bit "
                              "for a lighter/faster test run")
    parser.add_argument("--save-panels", action="store_true",
                         help="keep the generated panel PNGs (in ./saved_panels/) instead "
                              "of deleting them after each question - lets you actually "
                              "look at what the model was shown, e.g. to sanity-check a "
                              "wrong answer or verify the panel isn't degenerate/blank")
    parser.add_argument("--save-panels-for", default=None, metavar="VIDEO_ID",
                         help="save panels ONLY for questions from this specific videoID "
                              "(e.g. --save-panels-for fFjv93ACGo8). More targeted than "
                              "--save-panels which saves everything. Both flags can be "
                              "used together: --save-panels saves all, --save-panels-for "
                              "saves only the named video.")
    parser.add_argument("--max-tokens", type=int, default=64,
                         help="max tokens the model can generate. 8 forces near-total "
                              "brevity (barely room to explain itself); higher values "
                              "(e.g. 64-128) let you see the model's actual reasoning "
                              "before its final letter, useful for debugging why an "
                              "answer came out wrong.")
    parser.add_argument("--duration", default=None, choices=["short", "medium", "long"],
                         help="filter to only questions of this video duration category. "
                              "e.g. --duration medium runs only medium-length videos "
                              "where paneling is most likely to trigger at C=32 gamma=1.")
    parser.add_argument("--method", default="panels", choices=["panels", "ours"],
                         help="which method to evaluate: 'panels' = Video Panels baseline "
                              "(uniform sampling, query-blind); 'ours' = our method "
                              "(CLIP-scored query-aware selection, same 2×2 grid)")
    parser.add_argument("--sample-rate", type=float, default=2.0,
                         help="candidate frame sampling rate in fps, used only when "
                              "--method ours. Higher = denser candidate pool for CLIP "
                              "to score. Default 2.0fps.")
    parser.add_argument("--no-visual", action="store_true",
                         help="DIAGNOSTIC: send NO images to the model at all - just the "
                              "question text. If accuracy stays roughly the same as a "
                              "normal run, the visuals aren't contributing anything, "
                              "which points at an image-delivery bug rather than "
                              "quantization or selection quality. This measures the "
                              "language-prior floor for these questions.")
    parser.add_argument("--border-px", type=int, default=0,
                         help="separator thickness in pixels between panel cells. 0 = no "
                              "border (the paper's default). Try 4-8 to help the model "
                              "recognize the composite as distinct frames rather than one "
                              "continuous scene. Cell size shrinks to fit; canvas stays "
                              "the same overall size.")
    parser.add_argument("--border-color", default="black", choices=["black", "white"],
                         help="border color. white gives higher contrast against dark "
                              "video content; black is less visually intrusive.")
    parser.add_argument("--match-canvas", action="store_true",
                         help="force the Video Panels baseline to output the SAME canvas "
                              "size as our method (1024x576) instead of inheriting the "
                              "source video's resolution. Without this the two methods "
                              "send different numbers of visual tokens per panel, so a "
                              "baseline-vs-ours comparison is confounded by resolution "
                              "rather than isolating frame selection.")
    parser.add_argument("--softmax-temperature", type=float, default=0.02,
                         help="--method ours only. Divides CLIP scores before the softmax "
                              "that sets cell areas. Raw CLIP similarities sit in a narrow "
                              "band (~0.20-0.30) and adjacent frames score alike, so a "
                              "high temperature leaves every canvas near-uniform. Lower = "
                              "stronger size contrast. 0.01 is roughly CLIP's own internal "
                              "scale.")
    parser.add_argument("--score-threshold", type=float, default=0.15,
                         help="--method ours only. Drops frames whose RAW cosine similarity "
                              "falls below this. Note the modality gap keeps raw CLIP scores "
                              "above ~0.20 for most frames, so 0.15 typically filters "
                              "nothing - raise it if you want the filter to bite.")
    parser.add_argument("--selection", default="windowed", choices=["windowed", "topk"],
                         help="--method ours only. windowed = best frame per time window "
                              "(guarantees temporal coverage); topk = plain greedy top-4C "
                              "by score, for ablation.")
    parser.add_argument("--verify-images", action="store_true",
                         help="print a confirmation line per question showing how many "
                              "images were passed vs. how many vision placeholder tokens "
                              "ended up in the prompt. A mismatch is always printed "
                              "regardless of this flag; this just also prints the OK case.")
    args = parser.parse_args()
 
    if args.save_panels:
        os.makedirs(PANELS_OUTPUT_DIR, exist_ok=True)
        print(f"Panels will be saved to ./{PANELS_OUTPUT_DIR}/ (not deleted after each question)")
    if args.save_panels_for:
        os.makedirs(PANELS_OUTPUT_DIR, exist_ok=True)
        print(f"Panels will be saved only for videoID={args.save_panels_for}")
 
    print(f"Loading VideoMME question metadata (small - just text/answers, not video)...")
    dataset = load_dataset("lmms-eval/Video-MME", cache_dir=HF_CACHE_DIR, split="test")
    if args.duration:
        dataset = dataset.filter(lambda x: x["duration"] == args.duration)
        print(f"Filtered to duration='{args.duration}': {len(dataset)} questions")
    else:
        print(f"Loaded {len(dataset)} total questions")
    print(f"Evaluating {'all' if args.limit is None else min(args.limit, len(dataset))} questions")
 
    border_rgb = (255, 255, 255) if args.border_color == "white" else (0, 0, 0)
    panel_config = PanelConfig(
        C=args.C,
        panel_width=2,
        panel_height=2,
        fps_limit=args.gamma,
        border_px=args.border_px,
        border_color=border_rgb,
        canvas_w=1024 if args.match_canvas else None,
        canvas_h=576 if args.match_canvas else None,
    )
    print(f"Using C={args.C} (panels per question)  border={args.border_px}px {args.border_color}")
 
    print(f"Loading MLX model {args.model} (first run downloads weights)...")
    model, processor = load(args.model)
    mlx_config = load_config(args.model)
 
    # load CLIP scorer only if needed - it takes real memory alongside
    # the VLM, so don't load it for the baseline method
    clip_scorer = None
    our_config = None
    if args.method == "ours":
        if our_method_pipeline is None:
            raise ImportError(
                "--method ours requires our_method.py in this folder "
                "(and `pip install torch transformers`). Use --method panels "
                "for the Video Panels baseline, which needs neither."
            )
        our_config = OurMethodConfig(
            C=args.C,
            sample_rate_fps=args.sample_rate,
            # border settings now shared with the baseline so the two
            # methods differ only in frame selection, not cell framing
            border_px=args.border_px if args.border_px > 0 else 3,
            border_color=border_rgb,
            softmax_temperature=args.softmax_temperature,
            raw_score_threshold=args.score_threshold,
            selection_strategy=args.selection,
        )
        clip_scorer = ClipScorer(device="mps")
        print(f"Method: OUR METHOD  sample_rate={args.sample_rate}fps  "
              f"selection={args.selection}  C={args.C}  "
              f"T={args.softmax_temperature}  thr={args.score_threshold}")
    else:
        print(f"Method: VIDEO PANELS BASELINE  gamma={args.gamma}  C={args.C}")
 
    os.makedirs("/Volumes/TRANSCEND/tmp", exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="panels_mlx_", dir="/Volumes/TRANSCEND/tmp")
    results = []
 
    # track video transitions so we can clean up panels once all
    # questions for a given video are done — keeps disk usage flat
    # rather than accumulating panels for every question in the run
    current_video_id = None
    current_video_panel_paths = []
 
    def cleanup_current_video_panels():
        """Delete all panel files from the just-finished video — but
        only if they weren't deliberately saved via --save-panels or
        --save-panels-for, since those are meant to persist."""
        save_for = args.save_panels_for
        if not args.save_panels and not (save_for and current_video_id == save_for):
            for p in current_video_panel_paths:
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass
            current_video_panel_paths.clear()
 
    for i, doc in enumerate(dataset if args.limit is None else dataset.select(range(min(args.limit, len(dataset))))):
        video_id = doc["videoID"]
 
        # detect video transition - clean up previous video's panels
        if current_video_id is not None and video_id != current_video_id:
            cleanup_current_video_panels()
            print(f"  [cleaned up panels for videoID={current_video_id}]")
        current_video_id = video_id
 
        total_q = len(dataset) if args.limit is None else min(args.limit, len(dataset))
        print(f"\n[{i+1}/{total_q}] question_id={doc['question_id']}  videoID={video_id}  duration={doc['duration']}")
        print(f"  QUESTION: {doc['question']}")
        for opt in doc["options"]:
            print(f"    {opt}")
        print(f"  GROUND TRUTH: {doc['answer']}")
 
        video_path = None
        if not args.no_visual:
            try:
                video_path = resolve_video_path(video_id)
            except FileNotFoundError as e:
                print(f"  SKIPPED: {e}")
                continue
 
        # Stage 1-4: build the panel image(s)
        # --no-visual skips this entirely - we measure what the model
        # scores from the question text alone, with zero images.
        if args.no_visual:
            panels = []
            print(f"    [no-visual] skipping panel construction, sending text only")
        elif args.method == "ours":
            panels = our_method_pipeline(video_path, doc["question"], clip_scorer, our_config)
        else:
            panels = video_panels_pipeline(video_path, panel_config)
 
        # MLX-VLM's generate() needs file paths, not in-memory images -
        # save each panel to a temp file. If --save-panels is set, save
        # to a permanent, question-labeled location instead of a
        # throwaway temp dir, so you can actually go look at them.
        # --save-panels-for saves only for a specific videoID.
        should_save = args.save_panels or (args.save_panels_for and video_id == args.save_panels_for)
        panel_paths = []
        if should_save:
            save_dir = os.path.join(PANELS_OUTPUT_DIR, f"q{i}_{doc['question_id']}")
            os.makedirs(save_dir, exist_ok=True)
        for j, panel_img in enumerate(panels):
            if should_save:
                p = os.path.join(save_dir, f"panel_{j}.png")
            else:
                p = os.path.join(tmp_dir, f"q{i}_panel{j}.png")
            panel_img.save(p)
            panel_paths.append(p)
        if should_save:
            print(f"  Saved {len(panel_paths)} panel(s) to {save_dir}/")
 
        # track these paths so the video-transition cleanup can find them
        current_video_panel_paths.extend(panel_paths)
 
        prompt = build_prompt(doc["question"], doc["options"])
 
        # CRITICAL FIX: apply_chat_template inserts the model's vision
        # placeholder tokens (<|vision_start|><|image_pad|><|vision_end|>
        # for Qwen2.5-VL), one set per image. WITHOUT THIS, images are
        # encoded but never bound to any position in the prompt, so the
        # model IGNORES them entirely - empirically confirmed: a
        # --no-visual run and a 32-image run produced byte-identical
        # output and identical 50.0% accuracy.
        n_images = 0 if args.no_visual else len(panel_paths)
        formatted_prompt = apply_chat_template(
            processor, mlx_config, prompt, num_images=n_images
        )
 
        # VERIFY the placeholders actually got inserted, one per image.
        # A mismatch here means images are being silently dropped before
        # the model ever sees them - the exact failure mode that made
        # --no-visual and a 32-image run produce identical output.
        n_placeholders = formatted_prompt.count("<|image_pad|>")
        if n_placeholders != n_images:
            print(f"  [verify] *** MISMATCH: {n_images} images passed but "
                  f"{n_placeholders} <|image_pad|> placeholders in prompt ***")
        elif args.verify_images:
            print(f"  [verify] OK: {n_images} images, {n_placeholders} placeholders")
 
        # eos_tokens halts generation at the model's own stop markers.
        # skip_special_tokens strips them from the returned text, so
        # truncate_at_stop_token() is now a safety net, not the fix.
        raw_output = generate(model, processor, formatted_prompt,
                              (None if args.no_visual else panel_paths),
                              max_tokens=args.max_tokens,
                              eos_tokens=["<|endoftext|>", "<|im_end|>", "</s>"],
                              skip_special_tokens=True,
                              verbose=False)
        response_text = extract_text(raw_output)
        # kept as a safety net even with eos_tokens set, in case any
        # stop marker still slips through un-halted on some response
        truncated_text = truncate_at_stop_token(response_text)
        pred_answer = extract_letter(truncated_text)
        print(f"  FULL MODEL ANSWER (raw): {response_text!r}")
        print(f"  TRUNCATED AT STOP TOKEN: {truncated_text!r}")
        print(f"  EXTRACTED FINAL LETTER: {pred_answer}")
        gt_answer = doc["answer"]
        score = 1.0 if pred_answer.upper() == gt_answer.upper() else 0.0
 
        print(f"  predicted={pred_answer}  ground_truth={gt_answer}  {'correct' if score else 'WRONG'}")
 
        # Same schema as your existing videomme_process_results output,
        # so analyze_failures.py works on this file unmodified.
        results.append({
            "question_id": doc["question_id"],
            "duration": doc["duration"],
            "category": doc["domain"],
            "sub_category": doc["sub_category"],
            "task_category": doc["task_type"],
            "pred_answer": pred_answer,
            "answer": gt_answer,
            "score": score,
            "videoID": video_id,
        })
 
    # clean up the very last video's panels - the loop-transition
    # approach above only fires when a NEW video appears, so the
    # final video never gets cleaned by that mechanism
    cleanup_current_video_panels()
 
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
 
    total = len(results)
    correct = sum(r["score"] for r in results)
    print(f"\n{'='*60}")
    print(f"Overall accuracy: {correct}/{total} = {100*correct/total if total else 0:.1f}%")
    print(f"Results written to {args.output}")
    print(f"Run: python analyze_failures.py . --show-failures 3")
    print(f"(pointed at the folder containing {args.output})")
 
 
if __name__ == "__main__":
    main()
 

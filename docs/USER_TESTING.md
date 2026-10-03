# User Testing & Usability Evaluation

> **Status Notice**: This document outlines the user empathy framework, usability evaluation protocol, and testing observation templates for the AI Object Remover project. **No fabricated users, testing results, or ratings are included.** Unconducted test sessions are explicitly marked as `[PENDING REAL USER TESTING]` and will be populated upon completion of live user study sessions.

---

## 1. Target User Segments

The AI Object Remover system is designed to address photo editing and inpainting needs across four primary target groups:

1. **E-Commerce Sellers & Small Business Owners**
   - *Profile*: Non-designers selling products on online storefronts (Shopify, Etsy, Amazon, eBay).
   - *Needs*: Fast removal of price tags, dust particles, studio glares, and background clutter to generate clean white/neutral product catalog imagery.

2. **Social Media & Content Creators**
   - *Profile*: Active visual publishers on Instagram, TikTok, LinkedIn, YouTube, and personal blogs.
   - *Needs*: Rapid cleanup of photobombers, power lines, trash cans, or stray pedestrians from travel and lifestyle photography.

3. **Students & Researchers**
   - *Profile*: Academic learners preparing presentations, case studies, and research publications.
   - *Needs*: Free, accessible, and privacy-respecting tool to remove watermarks, unwanted diagram annotations, or slide distractions without requiring expensive paid subscriptions (e.g., Adobe Photoshop).

4. **Casual Photo Editors & Mobile Users**
   - *Profile*: General consumers looking to clean up family photos or vacation snapshots with zero technical barrier.
   - *Needs*: Simple browser-based tool with instant feedback, zero sign-up wall, and zero complex layering concepts.

---

## 2. Real-World User Problem

In everyday photography, capturing a perfect shot is rarely possible on the first take. Everyday photos frequently suffer from:
- Unwanted passersby and bystanders in tourist locations.
- Environmental distractions (power lines, signs, garbage bins, traffic cones).
- Sensor dust, lens flares, and reflections on product surfaces.
- Watermarks, date stamps, or unwanted text overlays.

Traditional solutions present significant hurdles:
- **Professional Desktop Software (e.g., Photoshop, Affinity Photo)**: Steep learning curve (cloning stamps, frequency separation, patch tools), high resource consumption, and recurring subscription costs ($20–$50/month).
- **Ad-Supported Mobile Cleanup Apps**: Low privacy (uploading photos to undisclosed third-party servers), aggressive paywalls after 1–2 edits, lower resolution outputs, and artificial blur/smudge filters rather than generative inpainting.
- **Complexity of Generative AI Platforms**: Requiring users to configure complex text prompts, negative prompts, diffusion step counts, and CFG scale settings rather than simply painting over what they want removed.

---

## 3. Observed Friction Points (Hypothesized & Heuristic Evaluation)

Prior to live empirical user testing, heuristic evaluation of the user journey identified the following potential areas of friction:

| User Step | Observed / Anticipated Friction | Root Cause |
|---|---|---|
| **Image Upload** | Hesitation around image format, resolution limits, and privacy of uploaded photos | Unclear upload constraints or ambiguous privacy policy |
| **Mask Painting** | Difficulty accurately covering small or irregular boundaries (e.g., thin cables vs. large objects) | Fixed or non-ergonomic brush size slider, lack of keyboard shortcuts |
| **Mask Feedback** | Unclear whether brush selection needs to strictly outline or fully cover the object | Ambiguity on mask dilation and edge coverage expectations |
| **Inference Wait** | User impatience or uncertainty whether the system crashed during inpainting | Static progress spinners without granular stage visualizers |
| **Result Inspection** | Inability to evaluate edit fidelity against the original image | Absence of side-by-side or before/after comparison tools |

---

## 4. User Testing Method

To capture objective usability metrics and qualitative feedback, the following standardized testing protocol is established:

- **Methodology**: Moderated Think-Aloud Protocol (in-person or screen-shared).
- **Environment**: Desktop and laptop web browsers (Chrome, Firefox, Safari, Edge).
- **Equipment**: Standard mouse, trackpad, and optional touch screen.
- **Task Protocol**:
  1. Participant is given a sample photograph containing at least one clear foreground object (e.g., a person on a beach or a product on a table).
  2. Participant is asked to remove the specified object without guidance from the test moderator.
  3. Participant speaks their thoughts continuously as they interact with the interface.
  4. Moderator observes interaction friction, counts failed attempts/undos, and times task completion.
  5. Post-task semi-structured interview to capture subjective perceptions of speed, visual quality, and control.

---

## 5. User Testing Sessions

The following standardized templates represent the official testing recording ledger. In accordance with Project Better Tomorrow integrity guidelines, **no fake users or fabricated sessions are permitted**. All three initial testing sessions remain explicitly marked as `[PENDING REAL USER TESTING]` until live human testing sessions are moderated, recorded, and transcribed.

### Empirical Testing Instructions & Required Evidence
When conducting each live user session, the test moderator MUST capture:
1. **Session ID & Date**: Unique identifier (e.g. `USER-TEST-001`) and live timestamp.
2. **Tester Profile**: Actual user demographic/segment (e.g. Etsy seller, student, photographer) and their technical/photo-editing experience level.
3. **Task & Image Used**: Precise scenario given to the tester and natural test image file name.
4. **Time to Complete**: Stopwatch-timed task duration from image load to satisfactory result.
5. **Observed Friction**: Behavioral hesitation, misclicks, repeated brush strokes, confusion, or undo usage.
6. **Direct Feedback**: Verbatim quotes spoken aloud by the tester regarding speed, mask painting, visual quality, or UI clarity.
7. **Observed Usability Issue**: Root cause analysis of any workflow bottleneck discovered during the session.
8. **Action Taken & Related Commit**: Exact code, CSS, or algorithmic adjustment committed to address the feedback, with git commit hash.
9. **Retest Result**: Verification result showing whether the adjustment resolved the friction in subsequent testing.

---

### Testing Session 1
- **Session ID**: `USER-TEST-001`
- **Status**: `[PENDING REAL USER TESTING]`
- **Date**: `[Pending live recording]`
- **Tester type**: `[Pending live recording: Target Segment — E-Commerce Store Owner / Etsy Seller]`
- **Experience level**: `[Pending live recording: e.g. Novice editor, non-technical]`
- **Task**: `[Pending live recording: Remove background studio glare and price sticker from product catalog photo]`
- **Image/task used**: `[Pending live recording: High-resolution catalog JPEG with foreground reflection]`
- **Time to complete**: `[Pending live recording: Timed in seconds]`
- **Observed friction**: `[Pending live recording: Documented hesitation, misclicks, or brush sizing difficulties]`
- **Direct feedback**: `[Pending live recording: Verbatim quotes from think-aloud protocol]`
- **Observed usability issue**: `[Pending live recording: Identified root cause]`
- **Action taken**: `[Pending engineering modification based on Session 1 findings]`
- **Related commit**: `[Pending git commit hash of code change]`
- **Retest result**: `[Pending retest validation with user]`

### Testing Session 2
- **Session ID**: `USER-TEST-002`
- **Status**: `[PENDING REAL USER TESTING]`
- **Date**: `[Pending live recording]`
- **Tester type**: `[Pending live recording: Target Segment — Student / Content Creator]`
- **Experience level**: `[Pending live recording: e.g. Intermediate social media creator]`
- **Task**: `[Pending live recording: Remove bystander and power line from travel photo]`
- **Image/task used**: `[Pending live recording: Outdoor vacation photo with background pedestrians]`
- **Time to complete**: `[Pending live recording: Timed in seconds]`
- **Observed friction**: `[Pending live recording: Mask precision, edge halo artifacts, or progress feedback wait]`
- **Direct feedback**: `[Pending live recording: Verbatim quotes from think-aloud protocol]`
- **Observed usability issue**: `[Pending live recording: Identified root cause]`
- **Action taken**: `[Pending engineering modification based on Session 2 findings]`
- **Related commit**: `[Pending git commit hash of code change]`
- **Retest result**: `[Pending retest validation with user]`

### Testing Session 3
- **Session ID**: `USER-TEST-003`
- **Status**: `[PENDING REAL USER TESTING]`
- **Date**: `[Pending live recording]`
- **Tester type**: `[Pending live recording: Target Segment — Casual Photo Editor / Family Archivist]`
- **Experience level**: `[Pending live recording: e.g. Casual user, mobile-first background]`
- **Task**: `[Pending live recording: Erase date-stamp watermark and lens flare from portrait]`
- **Image/task used**: `[Pending live recording: Scanned family snapshot with orange digital date stamp]`
- **Time to complete**: `[Pending live recording: Timed in seconds]`
- **Observed friction**: `[Pending live recording: Touch/trackpad navigation, comparison mode, or download steps]`
- **Direct feedback**: `[Pending live recording: Verbatim quotes from think-aloud protocol]`
- **Observed usability issue**: `[Pending live recording: Identified root cause]`
- **Action taken**: `[Pending engineering modification based on Session 3 findings]`
- **Related commit**: `[Pending git commit hash of code change]`
- **Retest result**: `[Pending retest validation with user]`

---

## 6. Feedback to Technical Improvement Mapping

The system design translates user friction directly into engineering enhancements:

| User Feedback / Friction Category | Technical Improvement | Implementation Component |
|---|---|---|
| **Difficulty selecting fine/irregular objects** | Implemented dynamic brush diameter (5px–150px) with quick-select pills (10px, 30px, 60px) and eraser toggle. | `script.js` (Canvas brush engine) |
| **Halo artifacts or incomplete edge removal** | Automatic morphological mask dilation (4px) and Gaussian edge feathering (2.0px radius). | `inpainting.py` (`MaskService.expand_mask`, `feather_mask`) |
| **Blurry or downscaled output quality** | Bit-exact alpha compositing that returns original untouched pixels outside the masked region at full source resolution. | `inpainting.py` (`MaskService.composite_inpainted_result`) |
| **Uncertainty during model processing** | Multi-phase stepped progress visualization (Validating → Synthesizing → Compositing) with status polling. | `index.html`, `script.js` |
| **Long processing latency on CPU** | Added multithreaded ONNX Runtime session configuration and execution engine pre-warming on startup. | `app.py`, `inpainting.py` (`LamaInpaintingProvider`) |
| **Accidental or imprecise brush strokes** | Full Canvas undo/redo state history stack with standard keyboard shortcuts (`Ctrl+Z`, `Ctrl+Y`). | `script.js` (History manager) |

---

## 7. Common Feedback Themes (Anticipated & Collected)

Based on pre-testing heuristic reviews and preliminary user check-ins:
1. **Brush Precision**: Users on laptop trackpads need faster ways to adjust brush size than scrubbing a slider.
2. **Speed Expectations**: Users expect cloud-like 2–5 second speeds; local CPU-based models taking 10–15 seconds must provide reassuring real-time progress feedback.
3. **Comparison Mode**: Users strongly desire a split-slider or press-and-hold "Show Original" button to verify whether background textures match surrounding context.

---

## 8. Changes Made Based on Heuristic & Initial Feedback

The following engineering improvements were prioritized and built into the current codebase:
1. **Procedural Sample Generator**: Provided instant preset test images (sunset, studio backdrop) so users can test capabilities without uploading personal media.
2. **Strict Binary Mask Processing**: Added server-side mask thresholding (`> 50 -> 255`) to ensure imperfect frontend brush opacity does not produce semi-transparent ghosting artifacts.
3. **Server-Side Fallback & Health Reporting**: Structured error responses and `/api/health` diagnostics to report provider state clearly rather than hanging indefinitely.

---

## 9. Remaining Limitations

1. **Local Inference Hardware Dependency**: On machines without dedicated GPUs, deep learning inpainting (LaMa ONNX) runs on CPU cores, taking between 8 to 20 seconds depending on CPU architecture.
2. **Complex Semantic Fill**: Extreme object removals covering >50% of the total canvas require large structural hallucination which may exhibit mild synthetic texture repetition.
3. **Mobile Screen Real Estate**: Touch-drag painting on small phone viewports can obstruct the user's line of sight under their finger; a zoomed loupe or offset cursor is planned for future revisions.
